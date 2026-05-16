# Step 1: Import libraries
# This code implements a simple data science pipeline using the California Housing dataset. It includes loading the dataset, splitting it into training and testing sets, training a Multi-layer Perceptron Regressor, evaluating the model's performance, and visualizing the results with actual vs predicted plots for both training and testing data.
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load California Housing dataset
# The California Housing dataset is a popular dataset for regression tasks. It contains information about various features of houses in California, such as median income, house age, and average number of rooms, along with the target variable, which is the median house value. 
housing = fetch_california_housing(as_frame=True)

# Features + target as DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Step 3: Define X and y
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 5: Train MLP Regressor
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),   # custom hyperparameter
    alpha=0.001,                   # custom hyperparameter
    learning_rate_init=0.001,      # custom hyperparameter
    early_stopping=True,
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

# Step 6: Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Step 7: Train Metrics
print("Train R2 Score:", r2_score(y_train, y_train_pred))
print("Train RMSE:", mean_squared_error(y_train, y_train_pred) ** 0.5)

# Step 8: Test Metrics
print("Test R2 Score:", r2_score(y_test, y_test_pred))
print("Test RMSE:", mean_squared_error(y_test, y_test_pred) ** 0.5)

# ---------------------------------------------------
# Step 9: Actual vs Predicted Plot — Train
# ---------------------------------------------------

plt.figure(figsize=(6, 4))

plt.scatter(y_train, y_train_pred, alpha=0.5)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted — Train")

plt.plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
)

plt.tight_layout()

plt.savefig(
    "figures/actual_vs_predicted_train.png"
)

plt.close()

# ---------------------------------------------------
# Step 10: Actual vs Predicted Plot — Test
# ---------------------------------------------------

plt.figure(figsize=(6, 4))

plt.scatter(y_test, y_test_pred, alpha=0.5)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted — Test")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
)

plt.tight_layout()

plt.savefig(
    "figures/actual_vs_predicted_test.png"
)

plt.close()

print("Plots saved successfully in figures folder.")