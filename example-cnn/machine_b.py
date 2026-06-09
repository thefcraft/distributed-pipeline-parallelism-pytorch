import torch
import torch.nn as nn
import torch.optim as optim
from src.model_wrapper import ModelWrapper
from src.server import create_app


class MnistCNN_MachineB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return self.fc2(x)


LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    import uvicorn

    model = MnistCNN_MachineB().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model_wrapper = ModelWrapper(
        model=model,
        optimizer=optimizer,
        device=DEVICE,
    )
    try:
        uvicorn.run(
            app=create_app(model_wrapper=model_wrapper),
            host="0.0.0.0",
            port=8002,
            access_log=False,
        )
    finally: # TODO: save model
        ...
