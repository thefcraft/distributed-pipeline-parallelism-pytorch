import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from uuid import UUID
from .models import BaseHeader, ForwardHeader
from dataclasses import dataclass


@dataclass
class BatchState:
    input: np.ndarray
    cpu_rng: torch.Tensor
    cuda_rng: torch.Tensor | None


class ModelWrapper:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.cuda_avaliable = torch.cuda.is_available()
        self.batch_store: dict[UUID, BatchState] = {}

    def forward(
        self, header: ForwardHeader, data: bytes
    ) -> tuple[ForwardHeader, bytes]:
        data = bytearray(data)
        dtype = np.dtype(header.dtype)
        arr = np.frombuffer(data, dtype=dtype)
        arr = arr.reshape(header.shape)

        if header.grad_req:
            self.batch_store[header.batch_key] = BatchState(
                input=arr,
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state() if self.cuda_avaliable else None,
            )  # NOTE: for gradient checkpointing
        tensor = torch.from_numpy(arr).to(device=self.device) # type: ignore
        with torch.no_grad():
            result: torch.Tensor = self.model(tensor)

        result_arr = result.detach().cpu().numpy()

        return ForwardHeader(
            batch_key=header.batch_key,
            shape=result_arr.shape,
            dtype=result_arr.dtype.str,
            grad_req=header.grad_req,
        ), result_arr.tobytes()

    def backward(
        self, header: BaseHeader, grad_data: bytes
    ) -> tuple[BaseHeader, bytes]:
        state = self.batch_store.pop(header.batch_key)
        tensor = state.input

        self.optimizer.zero_grad()
        grad_data = bytearray(grad_data)
        dtype = np.dtype(header.dtype)
        arr = np.frombuffer(grad_data, dtype=dtype)
        arr = arr.reshape(header.shape)
        grad = torch.from_numpy(arr).to(device=self.device) # type: ignore

        tensor = torch.from_numpy(tensor).to(device=self.device) # type: ignore
        tensor.requires_grad_()


        # NOTE: save current RNG state before we tamper with it
        current_cpu_rng = torch.get_rng_state()
        current_cuda_rng = torch.cuda.get_rng_state() if self.cuda_avaliable else None

        # NOTE: restore to forward-time state for recomputation
        torch.set_rng_state(state.cpu_rng)
        if state.cuda_rng is not None:
            torch.cuda.set_rng_state(state.cuda_rng)

        result: torch.Tensor = self.model(tensor)

        # NOTE: restore RNG back to where it was before this backward call
        torch.set_rng_state(current_cpu_rng)
        if current_cuda_rng is not None:
            torch.cuda.set_rng_state(current_cuda_rng)

        result.backward(grad) # type: ignore
        self.optimizer.step()

        result_grad = tensor.grad.detach().cpu().numpy()  # pyright: ignore[reportOptionalMemberAccess]
        return BaseHeader(
            batch_key=header.batch_key,
            shape=result_grad.shape,
            dtype=result_grad.dtype.str,
        ), result_grad.tobytes()

    def backward_root(self, header: BaseHeader, grad_data: bytes) -> None:
        state = self.batch_store.pop(header.batch_key)
        tensor = state.input

        self.optimizer.zero_grad()
        grad_data = bytearray(grad_data)
        dtype = np.dtype(header.dtype)
        arr = np.frombuffer(grad_data, dtype=dtype)
        arr = arr.reshape(header.shape)
        grad = torch.from_numpy(arr).to(device=self.device) # type: ignore

        tensor = torch.from_numpy(tensor).to(device=self.device) # type: ignore
        
        # NOTE: save current RNG state before we tamper with it
        current_cpu_rng = torch.get_rng_state()
        current_cuda_rng = torch.cuda.get_rng_state() if self.cuda_avaliable else None

        # NOTE: restore to forward-time state for recomputation
        torch.set_rng_state(state.cpu_rng)
        if state.cuda_rng is not None:
            torch.cuda.set_rng_state(state.cuda_rng)

        result: torch.Tensor = self.model(tensor)

        # NOTE: restore RNG back to where it was before this backward call
        torch.set_rng_state(current_cpu_rng)
        if current_cuda_rng is not None:
            torch.cuda.set_rng_state(current_cuda_rng)


        result.backward(grad) # type: ignore
        self.optimizer.step()
        return None
