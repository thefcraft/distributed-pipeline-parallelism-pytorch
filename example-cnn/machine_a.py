import torch
import torch.nn as nn
import torch.optim as optim
from src.model_wrapper import ModelWrapper
from src.server import create_app


class MnistCNN_MachineA(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # Bx1x28x28  => Bx32x26x26
        x = self.relu(self.conv2(x))  # Bx32x26x26 => Bx64x24x24
        x = self.flatten(x)
        return x


LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    import uvicorn

    model = MnistCNN_MachineA().to(DEVICE)
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
            port=8001,
            access_log=False,
        )
    finally:  # TODO: save model
        ...
