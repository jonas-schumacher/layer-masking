import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training import train_model

NUM_SAMPLES = 10000
WEIGHT_FEATURE_1 = 2
WEIGHT_FEATURE_2 = -3
WEIGHT_AUTOREGRESSIVE = 0.5

FEATURE1 = "feature1"
FEATURE2 = "feature2"
FEATURES = [FEATURE1, FEATURE2]
TARGET = "target"

PAST_HORIZON = 1
FORECAST_HORIZON = 3
WINDOW_STEP = 1

np.random.seed(42)

# Create initial timeseries
index_names = pd.MultiIndex.from_tuples(
    [("past", i) for i in range(PAST_HORIZON)]
    + [("future", i) for i in range(FORECAST_HORIZON)]
)

df = pd.DataFrame(
    columns=[FEATURE1, FEATURE2, TARGET],
    index=range(NUM_SAMPLES),
    dtype=float,
)

df[FEATURE1] = np.random.randn(NUM_SAMPLES)
df[FEATURE2] = np.random.randn(NUM_SAMPLES)
df[TARGET] = (
    WEIGHT_FEATURE_1 * df[FEATURE1]
    + WEIGHT_FEATURE_2 * df[FEATURE2]
    + 5 * np.random.randn(NUM_SAMPLES)
)

for row in range(1, len(df)):
    df.loc[row, TARGET] += WEIGHT_AUTOREGRESSIVE * df.loc[row - 1, TARGET]
df = df.iloc[1:, :]

df[TARGET].plot()
plt.show()

# Create training data samples:

window_size = PAST_HORIZON + FORECAST_HORIZON

samples = [
    df.iloc[i : i + window_size].set_index(index_names).copy(deep=True)
    for i in range(0, len(df) - window_size + 1, WINDOW_STEP)
]

features = pd.DataFrame([s.stack() for s in samples])

targets = pd.DataFrame([s.stack().loc[["future"], :, [TARGET]] for s in samples])

base_mask = pd.DataFrame(
    index=targets.columns,
    columns=features.columns,
    data=0.0,
)

# Use Case 1: Separate linear regression:
mask1 = base_mask.copy(deep=True)
for horizon in range(FORECAST_HORIZON):
    mask1.loc[("future", horizon, TARGET), ("future", horizon, FEATURES)] = 1.0

# Use Case 2: Share features across time
mask2 = base_mask.copy(deep=True)
mask2.loc[("future", slice(None), TARGET), (slice(None), slice(None), FEATURES)] = 1.0

# Use Case 3: Share features and allow autoregressive input from the past
mask3 = base_mask.copy(deep=True)
mask3.loc[("future", slice(None), TARGET), (slice(None), slice(None), FEATURES)] = 1.0
mask3.loc[("future", slice(None), TARGET), ("past", slice(None), TARGET)] = 1.0

for mask in [mask1, mask2, mask3]:
    training_results = train_model(
        features=features.values,
        targets=targets.values,
        mask=mask.values,
    )

    loss_timeseries = pd.DataFrame(training_results.loss_timeseries)
    loss_timeseries.columns = [f"Forecast t={col}" for col in loss_timeseries.columns]

    loss_timeseries.plot()
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.show()

    trained_weights = pd.DataFrame(
        training_results.trained_weights, index=mask.index, columns=mask.columns
    )

    print(mask.to_string())
    print((trained_weights * mask).to_string())

print()