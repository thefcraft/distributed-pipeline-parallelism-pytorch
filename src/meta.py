import torch
import asyncio
import aiohttp
import uvicorn
import numpy as np
from uuid import UUID
from pydantic import HttpUrl
from fastapi import APIRouter, Response, status
from contextlib import asynccontextmanager
from functools import partial
from uuid import uuid4
from urllib.parse import urljoin
from typing import TYPE_CHECKING
from .async_utils import AsyncDict
from .models import ServerConfig, SuccessJobResponse, BaseHeader, ForwardHeader

if not TYPE_CHECKING:
    from fastapi import FastAPI, Request, WebSocket
else:
    from fastapi import (
        Request as _Request,
        FastAPI as _FastAPI,
        WebSocket as _WebSocket,
    )
    from dataclasses import dataclass

    @dataclass()
    class State:
        """define all state type here."""

        batch_store: AsyncDict[UUID, np.ndarray]

    class FastAPI(_FastAPI):
        state: State  # pyright: ignore[reportIncompatibleVariableOverride]

    class WebSocket(_WebSocket):
        app: FastAPI  # pyright: ignore[reportIncompatibleMethodOverride]

    class Request(_Request):
        app: FastAPI  # pyright: ignore[reportIncompatibleMethodOverride]


router = APIRouter()


@router.get("/isalive", status_code=status.HTTP_204_NO_CONTENT)
async def isalive() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/forward", response_model=SuccessJobResponse)
async def forward(request: Request) -> SuccessJobResponse:
    header = ForwardHeader.model_validate_json(request.headers["x-header"])
    payload = await request.body()
    data = bytearray(payload)

    dtype = np.dtype(header.dtype)
    arr = np.frombuffer(data, dtype=dtype)
    arr = arr.reshape(header.shape)

    request.app.state.batch_store.set(header.batch_key, arr)

    return SuccessJobResponse(batch_key=header.batch_key)


@asynccontextmanager
async def lifespan(app: FastAPI, *, batch_store: AsyncDict[UUID, np.ndarray]):
    try:
        app.state.batch_store = batch_store
        yield
    finally:
        del app.state.batch_store


class Meta:
    session: aiohttp.ClientSession

    def __init__(
        self,
        device: torch.device,
        servers: list[HttpUrl],
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        assert len(servers) != 0
        forward_servers = servers[1:] + [HttpUrl(url=f"http://{host}:{port}")]
        backward_servers = [None] + servers[:-1]

        self.servers = servers
        self.server_configs = [
            ServerConfig(
                forward_server=forward,
                backward_server=backward,
            )
            for forward, backward in zip(
                forward_servers,
                backward_servers,
                strict=True,
            )
        ]

        self.batch_store: AsyncDict[UUID, np.ndarray] = AsyncDict()

        self.host = host
        self.port = port

        self.device = device

    @asynccontextmanager
    async def start(self):
        app = FastAPI(lifespan=partial(lifespan, batch_store=self.batch_store))
        app.include_router(router)

        async with aiohttp.ClientSession() as session:
            for server_url, config in zip(
                self.servers, self.server_configs, strict=True
            ):
                async with session.post(
                    urljoin(str(server_url), "config"),
                    json=config.model_dump(mode="json"),
                ) as resp:
                    resp.raise_for_status()
                    await resp.read()

                async with session.delete(
                    urljoin(str(server_url), "batch-store"),
                ) as resp:
                    resp.raise_for_status()
                    await resp.read()

            config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                access_log=False,
            )
            server = uvicorn.Server(config)
            server_task = asyncio.create_task(server.serve())
            while not server.started:
                await asyncio.sleep(0.01)
            try:
                self.session = session
                yield
            finally:
                server.should_exit = True
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass

    async def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[UUID, torch.Tensor]:
        uuid = uuid4()

        arr = x.detach().cpu().numpy()
        header = ForwardHeader(
            batch_key=uuid,
            shape=arr.shape,
            dtype=arr.dtype.str,
            grad_req=torch.is_grad_enabled(),
        )

        async with self.session.post(
            urljoin(
                str(self.servers[0]),
                "forward",
            ),
            data=arr.tobytes(),
            headers={
                "x-header": header.model_dump_json(),
            },
        ) as resp:
            resp.raise_for_status()
            resp = SuccessJobResponse.model_validate(await resp.json())
            assert resp.batch_key == uuid

        result_arr = await self.batch_store.pop(uuid)
        result_tensor = torch.from_numpy(result_arr).to(device=self.device) # type: ignore
        if torch.is_grad_enabled():
            result_tensor.requires_grad_()
        return uuid, result_tensor

    async def backward(
        self,
        batch_key: UUID,
        grad: torch.Tensor,
    ) -> None:
        arr = grad.detach().cpu().numpy()
        header = BaseHeader(
            batch_key=batch_key,
            shape=arr.shape,
            dtype=arr.dtype.str,
        )

        async with self.session.post(
            urljoin(str(self.servers[-1]), "backward"),
            data=arr.tobytes(),
            headers={
                "x-header": header.model_dump_json(),
            },
        ) as resp:
            resp.raise_for_status()
            resp = SuccessJobResponse.model_validate(await resp.json())
            assert resp.batch_key == batch_key
