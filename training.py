from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from model import CustomModel

LEARNING_RATE = 1.0
NUM_TRAIN_EPOCHS = 5

torch.manual_seed(42)


@dataclass
class TrainingResults:
    loss_timeseries: List[Dict[int, float]]
    trained_weights: np.ndarray


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> TrainingResults:

    torch_features = torch.tensor(features, dtype=torch.float32)
    torch_targets = torch.tensor(targets, dtype=torch.float32)
    torch_mask = torch.tensor(mask, dtype=torch.float32)

    model = CustomModel(
        in_features=torch_features.shape[1],
        out_features=torch_targets.shape[1],
        mask=torch_mask,
    )

    print(
        f"Masked weights before training: {model.masked_layer.weight.detach()[torch_mask==0.0]}"
    )

    loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.LBFGS(
        params=model.parameters(),
        lr=LEARNING_RATE,
    )

    def train_epoch() -> None:
        model.train()

        def closure() -> torch.float:
            optimizer.zero_grad()
            y_pred = model(torch_features)
            loss = loss_function(y_pred, torch_targets)
            loss.backward()
            return loss

        optimizer.step(closure)

    def evaluate_epoch() -> Dict[int, float]:
        model.eval()
        with torch.no_grad():
            y_pred = model(torch_features)
            metrics = {
                t: mean_absolute_error(
                    y_pred=y_pred.numpy()[:, t],
                    y_true=torch_targets.numpy()[:, t],
                )
                for t in range(torch_targets.shape[1])
            }
            return metrics

    loss_timeseries = [evaluate_epoch()]

    for current_epoch in range(NUM_TRAIN_EPOCHS):
        train_epoch()
        loss_timeseries.append(evaluate_epoch())

    trained_weights = model.masked_layer.weight.detach().numpy()

    training_results = TrainingResults(
        loss_timeseries=loss_timeseries, trained_weights=trained_weights
    )

    print(
        f"Masked weights after training: {model.masked_layer.weight.detach()[torch_mask==0.0]}"
    )

    return training_results
