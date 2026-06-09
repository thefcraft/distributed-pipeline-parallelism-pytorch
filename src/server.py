import asyncio
import aiohttp
from pydantic import ValidationError
from contextlib import asynccontextmanager
from fastapi import APIRouter, Response, HTTPException, status
from typing import TYPE_CHECKING
from functools import partial
from urllib.parse import urljoin
from .models import ServerConfig, SuccessJobResponse, ForwardHeader, BaseHeader
from .async_utils import AsyncMutex
from .model_wrapper import ModelWrapper


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

        server_config: ServerConfig | None
        model_wrapper: AsyncMutex[ModelWrapper]
        session: aiohttp.ClientSession

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

@router.post("/config", response_model=ServerConfig)
async def set_backward_server(request: Request, config: ServerConfig) -> ServerConfig:
    request.app.state.server_config = config
    return config


@router.get("/config", response_model=ServerConfig | None)
async def get_config(request: Request) -> ServerConfig | None:
    return request.app.state.server_config


@router.delete("/batch-store", status_code=status.HTTP_204_NO_CONTENT)
async def clear_batch_store(request: Request) -> None:
    async with request.app.state.model_wrapper.lock() as wrapper:
        wrapper.batch_store.clear()


@router.post("/forward", response_model=SuccessJobResponse)
async def forward(request: Request) -> SuccessJobResponse:
    if request.app.state.server_config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    try:
        header = ForwardHeader.model_validate_json(request.headers["x-header"])
    except (KeyError, ValidationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid header: {e}"
        )

    payload = await request.body()
    async with request.app.state.model_wrapper.lock() as model_wrapper:
        result_header, result_bytes = await asyncio.to_thread(
            model_wrapper.forward,
            header,
            payload,
        )
    async with request.app.state.session.post(
        urljoin(str(request.app.state.server_config.forward_server), "forward"),
        data=result_bytes,
        headers={
            "x-header": result_header.model_dump_json(),
        },
    ) as resp:
        resp.raise_for_status()
        return SuccessJobResponse.model_validate(await resp.json())

@router.post("/backward", response_model=SuccessJobResponse)
async def backward(request: Request) -> SuccessJobResponse:
    if request.app.state.server_config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    try:
        header = BaseHeader.model_validate_json(request.headers["x-header"])
    except (KeyError, ValidationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid header: {e}"
        )
    payload = await request.body()

    if request.app.state.server_config.backward_server is None:
        try:
            async with request.app.state.model_wrapper.lock() as model_wrapper:
                await asyncio.to_thread(
                    model_wrapper.backward_root,
                    header,
                    payload,
                )
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown batch_key: {header.batch_key}",
            )
        return SuccessJobResponse(batch_key=header.batch_key)

    try:
        async with request.app.state.model_wrapper.lock() as model_wrapper:
            result_header, result_bytes = await asyncio.to_thread(
                model_wrapper.backward,
                header,
                payload,
            )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown batch_key: {header.batch_key}",
        )

    async with request.app.state.session.post(
        urljoin(str(request.app.state.server_config.backward_server), "backward"),
        data=result_bytes,
        headers={
            "x-header": result_header.model_dump_json(),
        },
    ) as resp:
        resp.raise_for_status()
        return SuccessJobResponse.model_validate(await resp.json())


@asynccontextmanager
async def lifespan(app: FastAPI, *, model_wrapper: ModelWrapper):
    async with aiohttp.ClientSession() as session:
        try:
            app.state.session = session
            app.state.server_config = None
            app.state.model_wrapper = AsyncMutex(model_wrapper)
            yield
        finally:
            del app.state.session
            del app.state.server_config
            del app.state.model_wrapper


def create_app(model_wrapper: ModelWrapper) -> FastAPI:
    app = FastAPI(lifespan=partial(lifespan, model_wrapper=model_wrapper))
    app.include_router(router)
    return app
