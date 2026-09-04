from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_data(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Validate generic sequence classification data.

    Expected:
        X -> [samples, sequence_length, features]
        y -> [samples]
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")

    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")

    if X.ndim != 3:
        raise ValueError(
            "X must have shape "
            "[samples, sequence_length, features]."
        )

    if y.ndim != 1:
        raise ValueError(
            "y must have shape [samples]."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if len(X) == 0:
        raise ValueError("Dataset is empty.")

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "y contains NaN or infinite values."
        )


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """
    Convert numpy arrays into a PyTorch DataLoader.
    """

    validate_data(X, y)

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        probabilities = model(X_batch)

        probabilities = probabilities.squeeze(-1)

        loss = criterion(
            probabilities,
            y_batch,
        )

        loss.backward()

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate model loss without updating weights.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            probabilities = model(X_batch)

            probabilities = probabilities.squeeze(-1)

            loss = criterion(
                probabilities,
                y_batch,
            )

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    path: str | Path,
) -> None:
    """
    Save a model checkpoint.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    seed: int = 42,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """
    Generic LSTM training function.

    This function is completely dataset-independent.

    Parameters
    ----------
    X_train:
        Shape [samples, sequence_length, features]

    y_train:
        Shape [samples]

    X_validation:
        Shape [samples, sequence_length, features]

    y_validation:
        Shape [samples]
    """

    validate_data(
        X_train,
        y_train,
    )

    validate_data(
        X_validation,
        y_validation,
    )

    if X_train.shape[1:] != X_validation.shape[1:]:
        raise ValueError(
            "Training and validation feature dimensions "
            "must match."
        )

    set_seed(seed)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    model = model.to(device)

    train_loader = create_dataloader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = create_dataloader(
        X_validation,
        y_validation,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
    }

    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
            device,
        )

        history["train_loss"].append(
            train_loss
        )

        history["validation_loss"].append(
            validation_loss
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            best_epoch = epoch

            if checkpoint_path is not None:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    validation_loss=validation_loss,
                    path=checkpoint_path,
                )

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "device": str(device),
    }