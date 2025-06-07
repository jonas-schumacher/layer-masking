import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

NUM_SAMPLES = 100
WEIGHT_FEATURE_1 = 2
WEIGHT_FEATURE_2 = -3
WEIGHT_AUTOREGRESSIVE = 0.5
LEARNING_RATE = 0.2
NUM_TRAIN_EPOCHS = 20

np.random.seed(42)
torch.manual_seed(42)

from sklearn.metrics import mean_squared_error


class MaskedLinearLayer(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False,  # no need to use a bias
        )
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.linear(x, self.weight * self.mask, self.bias)
        return x


class MaskedLinearModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
    ) -> None:
        super(MaskedLinearModel, self).__init__()

        self.masked_linear_layer = MaskedLinearLayer(
            in_features=in_features,
            out_features=out_features,
            mask=mask,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.masked_linear_layer(x)
        return x


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    def train_epoch() -> None:
        model.train()

        def closure() -> torch.float:
            optimizer.zero_grad()
            y_pred = model(torch_features)
            loss = loss_function(y_pred, torch_targets)
            loss.backward()
            return loss

        optimizer.step(closure)

    def evaluate_epoch() -> float:
        model.eval()
        with torch.no_grad():
            y_pred = model(torch_features)
            metric = mean_squared_error(
                y_pred=y_pred.numpy(),
                y_true=torch_targets.numpy(),
            )
            return metric

    evaluate_epoch()

    history = []

    for current_epoch in range(NUM_TRAIN_EPOCHS):
        train_epoch()
        metric = evaluate_epoch()
        history.append(metric)

    return history


# DATASET

df = pd.DataFrame(
    data={
        "f1": np.random.randn(NUM_SAMPLES),
        "f2": np.random.randn(NUM_SAMPLES),
        "t1": np.ones(NUM_SAMPLES),
    },
    index=range(NUM_SAMPLES),
    dtype=float,
)

for row in range(1, len(df)):
    df.loc[row, "t1"] = (
        WEIGHT_FEATURE_1 * df.loc[row, "f1"]
        + WEIGHT_FEATURE_2 * df.loc[row, "f2"]
        + WEIGHT_AUTOREGRESSIVE * df.loc[row - 1, "t1"]
    )

df["t1"].plot()
plt.show()

features = df.loc[:, ["f1", "f2"]]
targets = df.loc[:, ["t1"]]

mask = torch.ones((targets.shape[1], features.shape[1]))

model = MaskedLinearModel(in_features=2, out_features=1, mask=mask)

torch_features = torch.tensor(features.values[1:], dtype=torch.float32)
torch_targets = torch.tensor(targets.values[1:], dtype=torch.float32)

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=LEARNING_RATE,
)

history = train_model(
    model=model,
    optimizer=optimizer,
)

plt.plot(history)
plt.show()

print(history)
