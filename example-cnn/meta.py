import asyncio
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.meta import Meta, HttpUrl
from torchvision import datasets, transforms  # pyright: ignore[reportMissingTypeStubs]
from tqdm import tqdm

# torch.random.manual_seed(42)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = "dataset"
MAX_BATCH_QUEUE = 48
BATCH_SIZE = 128
EPOCHS = 5

os.makedirs(DATASET_DIR, exist_ok=True)

train_ds = datasets.MNIST(
    DATASET_DIR,
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_ds = datasets.MNIST(
    DATASET_DIR,
    train=False,
    transform=transforms.ToTensor(),
)

train_loader: DataLoader[datasets.MNIST] = DataLoader(  # pyright: ignore[reportUnknownVariableType]
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader: DataLoader[datasets.MNIST] = DataLoader(  # pyright: ignore[reportUnknownVariableType]
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
criterion = nn.CrossEntropyLoss()


async def main_train_point(
    meta: Meta,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    batch_key, y_out = await meta.forward(x)
    loss: torch.Tensor = criterion(y_out, y)
    loss.backward()  # type: ignore
    await meta.backward(batch_key, y_out.grad)  # type: ignore
    return loss.detach().cpu().item()


async def train_loop(meta: Meta, pbar: tqdm) -> float: # type: ignore
    tasks: list[asyncio.Task[float]] = []

    running_loss = 0.0
    completed = 0

    for x, y in pbar:  # type: ignore
        task = asyncio.create_task(
            main_train_point(
                meta=meta,
                x=x,  # type: ignore
                y=y,  # type: ignore
            )
        )
        tasks.append(task)

        if len(tasks) >= MAX_BATCH_QUEUE:
            results = await asyncio.gather(*tasks)
            await meta.step()

            for loss in results:
                running_loss += loss
                completed += 1

            tasks = []
            pbar.set_postfix(loss=f"{running_loss / completed:.4f}")  # type: ignore

    if tasks:
        results = await asyncio.gather(*tasks)
        await meta.step()

        for loss in results:
            running_loss += loss
            completed += 1

    epoch_loss = running_loss / completed
    return epoch_loss


async def evaluate_loop(meta: Meta) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            _, logits = await meta.forward(x)

            loss: torch.Tensor = criterion(logits, y)

            total_loss += loss.item() * y.size(0)

            pred = logits.argmax(dim=1)

            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


async def main() -> None:
    meta = Meta(
        device=DEVICE,
        servers=[
            HttpUrl("http://127.0.0.1:8001"),
            HttpUrl("http://127.0.0.1:8002"),
        ],
        host="127.0.0.1",
        port=8000,
    )
    async with meta.start():
        for epoch in range(EPOCHS):
            pbar = tqdm(
                train_loader,
                desc=f"Training [{epoch + 1}/{EPOCHS}]",
            )
            epoch_loss = await train_loop(meta=meta, pbar=pbar)
            test_loss, test_acc = await evaluate_loop(meta)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"train_loss={epoch_loss:.6f} "
                f"test_loss={test_loss:.6f} "
                f"test_acc={100 * test_acc:.2f}%"
            )


if __name__ == "__main__":
    asyncio.run(main())
