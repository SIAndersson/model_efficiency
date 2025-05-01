"""
Invoice Payment Date Prediction
===============================
This script compares different models for predicting when clients will pay invoices:
1. Simple stats
2. Statistical model (linear regression)
3. Simple neural network
4. Complex neural network
5. Transformer model

It evaluates each model's performance, resource usage, and analyzes how predictions
correlate with client behavior profiles.
"""

import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import scipy.stats as stats
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import argparse

# --------------------------------

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

# Set seaborn style for plots
sns.set_theme(style="whitegrid", context="talk")

# --------------------------------
# Data Loading and Preprocessing
# --------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train models for invoice payment prediction.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    return parser.parse_args()

class DataProcessor:
    def __init__(self, data_path: str):
        """Initialize the data processor.

        Args:
            data_path: Path to the CSV file containing invoice data
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> pd.DataFrame:
        """Load the invoice dataset."""
        self.data = pd.read_csv(self.data_path)

        # Convert date columns to datetime
        self.data["due_date"] = pd.to_datetime(self.data["due_date"])
        self.data["payment_date"] = pd.to_datetime(self.data["payment_date"])

        # Create features from dates
        self.data["due_month"] = self.data["due_date"].dt.month
        self.data["due_day"] = self.data["due_date"].dt.day
        self.data["due_weekday"] = self.data["due_date"].dt.weekday

        # Calculate days_to_payment if not already in dataset
        if "days_to_payment" not in self.data.columns:
            self.data["days_to_payment"] = (
                self.data["payment_date"] - self.data["due_date"]
            ).dt.days

        # Check for missing or infinite values
        self.data = self.data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["days_to_payment"]
        )

        if "client_encoded" not in self.data.columns:
            self.data["client_encoded"] = (
                self.data["client"].astype("category").cat.codes
            )
            
        self.data['last_3_avg_days'] = self.data.groupby('client_encoded')['days_to_payment'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        
        # Fill with client-specific average when available, commented out cb its so slow
        for client in self.data['client_encoded'].unique():
            client_mask = self.data['client_encoded'] == client
            client_avg = self.data.loc[client_mask, 'last_3_avg_days'].mean()
            if not np.isnan(client_avg):
                self.data.loc[client_mask, 'last_3_avg_days'] = self.data.loc[client_mask, 'last_3_avg_days'].fillna(client_avg)
                
        #self.data = self.data.fillna(self.data.mean())  # Fill any remaining NaN values with mean

        return self.data

    def prepare_features(self) -> None:
        """Prepare features and targets for modeling."""
        # Features and target
        X = self.data[["client_encoded", "due_month", "due_day", "due_weekday", "last_3_avg_days"]]
        y = self.data["days_to_payment"]

        # Get unique clients
        unique_clients = self.data["client_encoded"].unique()

        # Initialize empty dataframes for train and test sets
        X_train_frames = []
        X_test_frames = []
        y_train_frames = []
        y_test_frames = []

        # For each client, split their data and add to respective collections
        for client in unique_clients:
            # Filter data for this client
            client_mask = self.data["client_encoded"] == client
            client_X = X[client_mask]
            client_y = y[client_mask]

            # Split this client's data
            if len(client_X) > 1:  # Need at least 2 samples to split
                X_train_client, X_test_client, y_train_client, y_test_client = (
                    train_test_split(client_X, client_y, test_size=0.2, random_state=42)
                )
            else:
                # If only one sample, put it in training
                X_train_client, X_test_client = client_X, pd.DataFrame()
                y_train_client, y_test_client = client_y, pd.Series(dtype=float)

            # Add to our collections
            X_train_frames.append(X_train_client)
            X_test_frames.append(X_test_client)
            y_train_frames.append(y_train_client)
            y_test_frames.append(y_test_client)

        # Concatenate all client-specific dataframes
        self.X_train = pd.concat(X_train_frames, axis=0)
        self.X_test = pd.concat(X_test_frames, axis=0)
        self.y_train = pd.concat(y_train_frames, axis=0)
        self.y_test = pd.concat(y_test_frames, axis=0)

        # Verify that all clients are represented in both train and test sets
        train_clients = set(self.X_train["client_encoded"].unique())
        test_clients = set(self.X_test["client_encoded"].unique())

        # Handle cases where some clients might be missing from test set
        missing_clients = train_clients - test_clients
        if missing_clients:
            # For each missing client, move some samples from train to test
            for client in missing_clients:
                client_mask = self.X_train["client_encoded"] == client
                client_indices = self.X_train[client_mask].index

                if len(client_indices) > 0:
                    # Take at least one sample to move to test set
                    move_idx = client_indices[0]

                    # Move from train to test
                    self.X_test = pd.concat([self.X_test, self.X_train.loc[[move_idx]]])
                    self.y_test = pd.concat([self.y_test, self.y_train.loc[[move_idx]]])

                    # Remove from training set
                    self.X_train = self.X_train.drop(move_idx)
                    self.y_train = self.y_train.drop(move_idx)

        assert len(self.y_test) == len(self.X_test)
        assert len(self.y_train) == len(self.X_train)

    def get_processed_data(self) -> Tuple:
        """Return the processed training and testing data."""
        return (
            self.y_train,
            self.y_test,
            self.X_train,
            self.X_test,
        )


class ClientPaymentDataset(Dataset):
    """Custom dataset for client payment history."""

    def __init__(self, X, y, client_history_length=20):
        """
        Args:
            X: DataFrame with features (client_encoded, due_month, due_day, due_weekday)
            y: Series with target values (days_to_payment)
            client_history_length: Number of past invoices to include in history
        """
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.client_history_length = client_history_length

        # Group by client to get history
        self.client_data = {}
        for client_id in self.X["client_encoded"].unique():
            client_mask = self.X["client_encoded"] == client_id
            client_X = self.X[client_mask].sort_values(by=["due_month", "due_day"])
            client_y = self.y[client_mask].loc[client_X.index]

            # Store client history
            self.client_data[client_id] = {"X": client_X.values, "y": client_y.values}

        # Prepare indices pointing to valid samples (those with sufficient history)
        self.valid_indices = []
        for client_id, data in self.client_data.items():
            # For each invoice after the first client_history_length
            if len(data["y"]) > self.client_history_length:
                for i in range(self.client_history_length, len(data["y"])):
                    self.valid_indices.append((client_id, i))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        client_id, invoice_idx = self.valid_indices[idx]
        client_data = self.client_data[client_id]

        # Get historical data leading up to this invoice
        history_start = invoice_idx - self.client_history_length
        history_end = invoice_idx

        # Features: client_id, due_month, due_day, due_weekday
        current_features = client_data["X"][invoice_idx].copy()
        history_features = client_data["X"][history_start:history_end].copy()
        history_targets = client_data["y"][history_start:history_end].copy()

        # Target: days_to_payment
        target = client_data["y"][invoice_idx]

        return {
            "client_id": torch.tensor(current_features[0], dtype=torch.long),
            "current_features": torch.tensor(current_features[1:], dtype=torch.float),
            "history_features": torch.tensor(history_features, dtype=torch.float),
            "history_targets": torch.tensor(history_targets, dtype=torch.float),
            "target": torch.tensor(target, dtype=torch.float),
        }


# --------------------------------
# Model Classes
# --------------------------------


class BaseModel:
    """Base class for all models."""

    def __init__(self, name: str):
        """Initialize the base model.

        Args:
            name: Model name for identification
        """
        self.name = name
        self.model = None
        self.training_time = 0
        self.memory_usage = 0
        self.gpu_memory_usage = 0
        self.metrics = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")

    def track_resources(func):
        """Decorator to track resource usage during training."""

        def wrapper(self, *args, **kwargs):
            # Get initial resource usage
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
            start_time = time.time()

            # Track GPU memory if available
            start_gpu_memory = 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                start_gpu_memory = torch.mps.driver_allocated_memory() / (
                    1024 * 1024
                )  # MB

            # Run the function (train)
            result = func(self, *args, **kwargs)

            # Get final resource usage
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # Track GPU memory if available
            end_gpu_memory = 0
            if torch.cuda.is_available():
                end_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                self.gpu_memory_usage = torch.cuda.max_memory_allocated() / (
                    1024 * 1024
                )  # Peak usage
            elif torch.backends.mps.is_available():
                end_gpu_memory = torch.mps.driver_allocated_memory() / (1024 * 1024)
                self.gpu_memory_usage = end_gpu_memory - start_gpu_memory
            else:
                self.gpu_memory_usage = end_gpu_memory - start_gpu_memory

            # Store metrics
            self.training_time = end_time - start_time
            self.memory_usage = end_memory - start_memory

            return result

        return wrapper

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate the model and return performance metrics."""
        y_pred = self.predict(X_test)

        # Calculate metrics
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
            
        # See if we have NaN
        if np.isnan(y_test).any():
            print(f"Warning: {len(y_test[np.isnan(y_test)])} NaN values found in test data out of {len(y_test)}.")
            print(y_test)
        if np.isnan(y_pred).any():
            print(f"Warning: {len(y_pred[np.isnan(y_pred)])} NaN values found in prediction data out of {len(y_pred)}.")
            print(y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Store and return metrics
        self.metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "training_time": self.training_time,
            "memory_usage": self.memory_usage,
            "gpu_memory_usage": self.gpu_memory_usage,
        }

        return self.metrics

    def predict(self, X):
        """Predict using the trained model."""
        raise NotImplementedError("Subclasses must implement this method")

    def save_model(self, path: str):
        """Save the trained model."""
        raise NotImplementedError("Subclasses must implement this method")


class StatisticalPaymentPredictor(BaseModel):
    """
    Statistical model to predict payment timing based on historical data.
    Uses descriptive statistics rather than ML approaches.

    The model calculates the difference between due date and actual payment date
    for each client, then uses statistical measures to predict future payments.
    """

    def __init__(self, prediction_method: str = "mean", confidence_level: float = 0.95):
        """
        Initialize the statistical payment predictor.

        Args:
            prediction_method: Method to use for prediction, one of:
                'mean': Use the mean of past payment delays
                'median': Use the median of past payment delays
                'mode': Use the most common payment delay
                'weighted_mean': Use recency-weighted mean of past payment delays
                'confidence': Use the upper bound of the confidence interval
            confidence_level: Confidence level for interval predictions (0-1)
        """
        super().__init__(name=f"StatisticalPaymentPredictor_{prediction_method}")
        self.prediction_method = prediction_method
        self.confidence_level = confidence_level
        self.client_payment_history = defaultdict(list)
        self.client_statistics = {}

    def _calculate_payment_delay(self, due_date, payment_date):
        """
        Calculate the difference in days between due date and payment date.
        Positive values indicate late payments, negative values indicate early payments.
        """
        return (payment_date - due_date).days

    def _calculate_client_statistics(self, client_id: str) -> Dict:
        """
        Calculate statistical measures for a client's payment history.
        """
        delays = self.client_payment_history[client_id]

        if not delays:
            return {
                "count": 0,
                "mean": 0,
                "median": 0,
                "mode": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "q25": 0,
                "q75": 0,
                "confidence_interval": (0, 0),
            }

        # Calculate basic statistics
        mean_delay = np.mean(delays)
        median_delay = np.median(delays)
        std_delay = np.std(delays, ddof=1) if len(delays) > 1 else 0

        # Find mode (most common payment delay)
        unique_values, counts = np.unique(delays, return_counts=True)
        mode_delay = unique_values[np.argmax(counts)]

        # Calculate percentiles
        min_delay = np.min(delays)
        max_delay = np.max(delays)
        q25 = np.percentile(delays, 25)
        q75 = np.percentile(delays, 75)

        # Calculate confidence interval for the mean
        if len(delays) > 1:
            t_value = stats.t.ppf((1 + self.confidence_level) / 2, len(delays) - 1)
            margin_error = t_value * (std_delay / np.sqrt(len(delays)))
            confidence_interval = (mean_delay - margin_error, mean_delay + margin_error)
        else:
            confidence_interval = (mean_delay, mean_delay)

        return {
            "count": len(delays),
            "mean": mean_delay,
            "median": median_delay,
            "mode": mode_delay,
            "std": std_delay,
            "min": min_delay,
            "max": max_delay,
            "q25": q25,
            "q75": q75,
            "confidence_interval": confidence_interval,
        }

    def _calculate_weighted_mean(self, client_id: str) -> float:
        """
        Calculate recency-weighted mean where more recent payments have higher weights.
        """
        delays = self.client_payment_history[client_id]
        if not delays:
            return 0

        # Weight by position (more recent = higher weight)
        weights = np.arange(1, len(delays) + 1)
        weighted_mean = np.average(delays, weights=weights)
        return weighted_mean

    @BaseModel.track_resources
    def train(self, data: pd.DataFrame) -> Dict:
        """
        Train the model using historical payment data.

        Args:
            data: DataFrame with columns:
                - client_id: Identifier for the client
                - due_date: Date when payment was due
                - payment_date: Date when payment was actually made
                - days_to_payment: Difference in days between due date and payment date

        Returns:
            Dictionary with training statistics
        """
        # Ensure dates are datetime objects
        for date_col in ["due_date", "payment_date"]:
            if date_col in data.columns and not pd.api.types.is_datetime64_dtype(
                data[date_col]
            ):
                data[date_col] = pd.to_datetime(data[date_col])

        # Calculate payment delays for all records
        data["payment_delay"] = data["days_to_payment"]

        # Group by client and collect payment delays
        for client_id, group in data.groupby("client_encoded"):
            # Sort by payment date to maintain chronological order
            sorted_group = group.sort_values("payment_date")
            self.client_payment_history[client_id] = sorted_group[
                "payment_delay"
            ].tolist()

        # Calculate statistics for each client
        for client_id in self.client_payment_history:
            self.client_statistics[client_id] = self._calculate_client_statistics(
                client_id
            )
            # Add weighted mean
            self.client_statistics[client_id]["weighted_mean"] = (
                self._calculate_weighted_mean(client_id)
            )

        # Return simple stats about the training
        """return {
            "num_clients": len(self.client_payment_history),
            "total_records": len(data),
            "avg_records_per_client": len(data) / len(self.client_payment_history)
            if self.client_payment_history
            else 0,
        }"""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict payment delays for new invoices.

        Args:
            X: DataFrame with at least a 'client_id' column

        Returns:
            Array of predicted payment delays in days
        """
        predictions = []

        for _, row in X.iterrows():
            client_id = row["client_encoded"]

            # Use default values for new clients
            if client_id not in self.client_statistics:
                predictions.append(0)  # Default: assume on-time payment for new clients
                continue

            stats = self.client_statistics[client_id]

            # Make prediction based on selected method
            if self.prediction_method == "mean":
                predictions.append(stats["mean"])
            elif self.prediction_method == "median":
                predictions.append(stats["median"])
            elif self.prediction_method == "mode":
                predictions.append(stats["mode"])
            elif self.prediction_method == "weighted_mean":
                predictions.append(stats["weighted_mean"])
            elif self.prediction_method == "confidence":
                # Use upper bound of confidence interval (conservative estimate)
                predictions.append(stats["confidence_interval"][1])
            else:
                # Default to mean
                predictions.append(stats["mean"])

        return np.array(predictions)

    def predict_with_details(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict payment delays with additional statistical details.

        Args:
            X: DataFrame with at least a 'client_id' column

        Returns:
            DataFrame with predictions and statistical details
        """
        results = []

        for _, row in X.iterrows():
            client_id = row["client_encoded"]

            # Initialize with row data
            result = row.to_dict()

            # Add prediction
            result["predicted_delay"] = self.predict(pd.DataFrame([row]))[0]

            # Add statistical information
            if client_id in self.client_statistics:
                stats = self.client_statistics[client_id]
                result.update(
                    {
                        "history_count": stats["count"],
                        "mean_delay": stats["mean"],
                        "median_delay": stats["median"],
                        "mode_delay": stats["mode"],
                        "std_delay": stats["std"],
                        "min_delay": stats["min"],
                        "max_delay": stats["max"],
                        "confidence_interval_lower": stats["confidence_interval"][0],
                        "confidence_interval_upper": stats["confidence_interval"][1],
                    }
                )
            else:
                # Default values for new clients
                result.update(
                    {
                        "history_count": 0,
                        "mean_delay": None,
                        "median_delay": None,
                        "mode_delay": None,
                        "std_delay": None,
                        "min_delay": None,
                        "max_delay": None,
                        "confidence_interval_lower": None,
                        "confidence_interval_upper": None,
                    }
                )

            results.append(result)

        return pd.DataFrame(results)

    def get_client_statistics(self, client_id: str = None) -> Union[Dict, pd.DataFrame]:
        """
        Get statistics for a specific client or all clients.

        Args:
            client_id: Specific client ID (if None, returns all clients)

        Returns:
            Dictionary of statistics for the requested client or DataFrame for all clients
        """
        if client_id is not None:
            return self.client_statistics.get(client_id, {})

        # Convert to DataFrame for all clients
        stats_df = pd.DataFrame(
            [
                {
                    "client_encoded": client_id,
                    **{k: v for k, v in stats.items() if k != "confidence_interval"},
                    "confidence_interval_lower": stats["confidence_interval"][0],
                    "confidence_interval_upper": stats["confidence_interval"][1],
                }
                for client_id, stats in self.client_statistics.items()
            ]
        )

        return stats_df

    def save_model(self, path: str):
        """
        Save the trained model to a file.

        Args:
            path: File path to save the model
        """
        model_data = {
            "name": self.name,
            "prediction_method": self.prediction_method,
            "confidence_level": self.confidence_level,
            "client_payment_history": dict(self.client_payment_history),
            "client_statistics": self.client_statistics,
            "metrics": self.metrics,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, path: str) -> "StatisticalPaymentPredictor":
        """
        Load a trained model from a file.

        Args:
            path: File path to load the model from

        Returns:
            Loaded StatisticalPaymentPredictor instance
        """
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        model = cls(
            prediction_method=model_data["prediction_method"],
            confidence_level=model_data["confidence_level"],
        )

        model.name = model_data["name"]
        model.client_payment_history = defaultdict(
            list, model_data["client_payment_history"]
        )
        model.client_statistics = model_data["client_statistics"]
        model.metrics = model_data["metrics"]

        return model


class StatisticalModel(BaseModel):
    """Linear regression model."""

    def __init__(self):
        """Initialize the statistical model."""
        super().__init__(name="Statistical Model (Linear Regression)")
        self.model = LinearRegression()

    @BaseModel.track_resources
    def train(self, X_train, y_train, epochs=None, batch_size=None):
        """Train the linear regression model."""
        # Make sure training data doesn't contain NaN or infinity
        if np.isnan(X_train.values).any() or np.isinf(X_train.values).any():
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        if isinstance(y_train, pd.Series):
            y_train_clean = y_train.replace([np.inf, -np.inf], np.nan).dropna()
        else:
            y_train_clean = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

        self.model.fit(X_train, y_train_clean)
        return self

    def predict(self, X):
        """Make predictions using the trained model."""
        # Ensure X doesn't contain NaN or infinity
        if np.isnan(X.values).any() or np.isinf(X.values).any():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X)

    def save_model(self, path: str = "statistical_model.joblib"):
        """Save the model to disk."""
        joblib.dump(self.model, path)


class SimpleNeuralNetworkModule(nn.Module):
    """PyTorch module for simple neural network."""

    def __init__(self, input_dim):
        """Initialize the neural network layers.

        Args:
            input_dim: Dimension of input features
        """
        super(SimpleNeuralNetworkModule, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x


class SimpleNeuralNetwork(BaseModel):
    """Simple neural network with few layers."""

    def __init__(self):
        """Initialize the simple neural network."""
        super().__init__(name="Simple Neural Network")
        self.early_stop_patience = 10
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None

    def _init_weights(self, m):
        """Initialize weights for the neural network layers."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build_model(self, input_dim):
        """Build the neural network architecture."""
        model = SimpleNeuralNetworkModule(input_dim)
        # model.apply(self._init_weights)
        return model

    def _early_stopping(self, val_loss):
        """Implement early stopping logic."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False, True  # Don't stop, save model
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stop_patience:
                return True, False  # Stop, don't save model
            return False, False  # Don't stop, don't save model

    @BaseModel.track_resources
    def train(
        self, X_train, y_train, lr=1e-3, epochs=1, batch_size=256, validation_split=0.2
    ):
        """Train the simple neural network."""
        # Handle sparse matrix
        if sp.issparse(X_train):
            X_train = X_train.toarray()

        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)

        if isinstance(y_train, pd.Series):
            y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        else:
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

        # Stratified sampling to ensure every class of client_encoded is represented
        unique_clients = X_train["client_encoded"].unique()
        train_indices = []
        val_indices = []

        for client in unique_clients:
            client_mask = X_train["client_encoded"] == client
            client_indices = np.where(client_mask)[0]

            # Split indices for this client
            val_split = int(len(client_indices) * validation_split)
            val_indices.extend(client_indices[:val_split])
            train_indices.extend(client_indices[val_split:])

        # Create stratified train and validation sets
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        train_dataset = TensorDataset(
            X_train_tensor[train_indices], y_train_tensor[train_indices]
        )
        val_dataset = TensorDataset(
            X_train_tensor[val_indices], y_train_tensor[val_indices]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Build the model and move to device
        self.model = self._build_model(X_train.shape[1])
        self.model.to(self.device)

        # Define loss function and optimizer
        criterion = nn.HuberLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # Training loop
        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    pbar.set_postfix({"loss": loss.item()})

            train_loss = train_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

                val_loss = val_loss / len(val_loader.dataset)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            stop, save_model = self._early_stopping(val_loss)
            if save_model:
                self.best_model_state = self.model.state_dict().copy()
            if stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model if early stopping was triggered
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.plot_history()

        return self

    def predict(self, X):
        """Make predictions using the trained model."""
        # Handle sparse matrix
        if sp.issparse(X):
            X = X.toarray()

        self.model.eval()
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy().flatten()

    def save_model(self, path: str = "simple_nn_model.pt"):
        """Save the model to disk."""
        torch.save(self.model.state_dict(), path)

    def plot_history(self):
        """Plot training and validation loss history."""
        fig, ax = plt.subplots()
        sns.lineplot(self.history["train_loss"], label="Train Loss", ax=ax)
        sns.lineplot(self.history["val_loss"], label="Validation Loss", ax=ax)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        fig.suptitle("Training and Validation Loss History")
        plt.tight_layout()
        plt.savefig("simple_nn_history.png", bbox_inches="tight")
        plt.show()


class ComplexNeuralNetworkModule(nn.Module):
    """PyTorch module for complex neural network."""

    def __init__(self, input_dim):
        """Initialize the neural network layers.

        Args:
            input_dim: Dimension of input features
        """
        super(ComplexNeuralNetworkModule, self).__init__()

        # Layer 1
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        # Layer 2
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        # Layer 3
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        # Layer 4
        self.layer4 = nn.Linear(64, 32)

        # Output layer
        self.output = nn.Linear(32, 1)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network."""
        # Layer 1
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        # Layer 4
        x = self.layer4(x)
        x = self.relu(x)

        # Output
        x = self.output(x)

        return x


class ComplexNeuralNetwork(BaseModel):
    """Complex neural network with more layers and regularization."""

    def __init__(self):
        """Initialize the complex neural network."""
        super().__init__(name="Complex Neural Network")
        self.early_stop_patience = 15
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None

    def _init_weights(self, m):
        """Initialize weights for the neural network layers."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build_model(self, input_dim):
        """Build the neural network architecture."""
        model = ComplexNeuralNetworkModule(input_dim)
        # model.apply(self._init_weights)
        return model

    def _early_stopping(self, val_loss):
        """Implement early stopping logic."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False, True  # Don't stop, save model
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stop_patience:
                return True, False  # Stop, don't save model
            return False, False  # Don't stop, don't save model

    @BaseModel.track_resources
    def train(
        self, X_train, y_train, lr=5e-4, epochs=1, batch_size=256, validation_split=0.2
    ):
        """Train the complex neural network."""
        # Handle sparse matrix
        if sp.issparse(X_train):
            X_train = X_train.toarray()

        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)

        if isinstance(y_train, pd.Series):
            y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        else:
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

        # Stratified sampling to ensure every class of client_encoded is represented
        unique_clients = X_train["client_encoded"].unique()
        train_indices = []
        val_indices = []

        for client in unique_clients:
            client_mask = X_train["client_encoded"] == client
            client_indices = np.where(client_mask)[0]

            # Split indices for this client
            val_split = int(len(client_indices) * validation_split)
            val_indices.extend(client_indices[:val_split])
            train_indices.extend(client_indices[val_split:])

        # Create stratified train and validation sets
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        train_dataset = TensorDataset(
            X_train_tensor[train_indices], y_train_tensor[train_indices]
        )
        val_dataset = TensorDataset(
            X_train_tensor[val_indices], y_train_tensor[val_indices]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Build the model and move to device
        self.model = self._build_model(X_train.shape[1])
        self.model.to(self.device)

        # Define loss function and optimizer
        criterion = nn.HuberLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # Training loop
        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    pbar.set_postfix({"loss": loss.item()})

            train_loss = train_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

                val_loss = val_loss / len(val_loader.dataset)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            stop, save_model = self._early_stopping(val_loss)
            if save_model:
                self.best_model_state = self.model.state_dict().copy()
            if stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model if early stopping was triggered
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.plot_history()

        return self

    def predict(self, X):
        """Make predictions using the trained model."""
        # Handle sparse matrix
        if sp.issparse(X):
            X = X.toarray()

        self.model.eval()
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy().flatten()

    def save_model(self, path: str = "complex_nn_model.pt"):
        """Save the model to disk."""
        torch.save(self.model.state_dict(), path)

    def plot_history(self):
        """Plot training and validation loss history."""
        fig, ax = plt.subplots()
        sns.lineplot(self.history["train_loss"], label="Train Loss", ax=ax)
        sns.lineplot(self.history["val_loss"], label="Validation Loss", ax=ax)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        fig.suptitle("Training and Validation Loss History")
        plt.tight_layout()
        plt.savefig("complex_nn_history.png", bbox_inches="tight")
        plt.show()


class ClientPaymentTransformer(nn.Module):
    """Transformer-based model for client payment prediction without requiring history dataset."""

    def __init__(
        self,
        input_dim,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super(ClientPaymentTransformer, self).__init__()

        # Feature processing
        self.feature_projection = nn.Linear(input_dim, embed_dim)

        # Transformer encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x):
        # Project features to embedding dimension
        x = self.feature_projection(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)

        # Final prediction
        x = self.output_layer(x)

        return x.squeeze(-1)


class ClientPaymentPredictionModel(BaseModel):
    """Model for predicting client payment timing using standard data format."""

    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        """
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__(name="ClientPaymentTransformer")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Early stopping parameters
        self.early_stop_patience = 15
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None

    def _early_stopping(self, val_loss):
        """Implement early stopping logic."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False, True  # Don't stop, save model
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stop_patience:
                return True, False  # Stop, don't save model
            return False, False  # Don't stop, don't save model

    @BaseModel.track_resources
    def train(
        self,
        X_train,
        y_train,
        lr=1e-4,
        epochs=1,
        batch_size=256,
        validation_split=0.2,
        optimizer=None,
    ):
        """Train the transformer using the same interface as SimpleNN and ComplexNN."""
        # Handle sparse matrix
        if sp.issparse(X_train):
            X_train = X_train.toarray()

        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)

        if isinstance(y_train, pd.Series):
            y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        else:
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

        # Stratified sampling to ensure every class of client_encoded is represented
        unique_clients = X_train["client_encoded"].unique()
        train_indices = []
        val_indices = []

        for client in unique_clients:
            client_mask = X_train["client_encoded"] == client
            client_indices = np.where(client_mask)[0]

            # Split indices for this client
            val_split = int(len(client_indices) * validation_split)
            val_indices.extend(client_indices[:val_split])
            train_indices.extend(client_indices[val_split:])

        # Create stratified train and validation sets
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        train_dataset = TensorDataset(
            X_train_tensor[train_indices], y_train_tensor[train_indices]
        )
        val_dataset = TensorDataset(
            X_train_tensor[val_indices], y_train_tensor[val_indices]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Build the model and move to device
        self.model = ClientPaymentTransformer(
            input_dim=X_train.shape[1],
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.model.to(self.device)

        # Define loss function and optimizer
        criterion = nn.HuberLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        self.history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.squeeze(-1))

                    # Backward pass and optimize
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    pbar.set_postfix({"loss": loss.item()})

            train_loss = train_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.squeeze(-1))
                    val_loss += loss.item() * inputs.size(0)

                val_loss = val_loss / len(val_loader.dataset)

            # Update learning rate
            scheduler.step(val_loss)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            stop, save_model = self._early_stopping(val_loss)
            if save_model:
                self.best_model_state = self.model.state_dict().copy()
            if stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model if early stopping was triggered
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        self.plot_history()

        return self

    def predict(self, X, batch_size=1024):
        """Make predictions using the trained model."""

        self.model.eval()
        predictions = []

        # Convert to PyTorch tensor
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.FloatTensor(X.values)
        else:
            X_tensor = torch.FloatTensor(X)

        # Process in batches to avoid memory issues
        for i in range(0, X_tensor.shape[0], batch_size):
            batch = X_tensor[i : i + batch_size].to(self.device)
            with torch.no_grad():
                batch_predictions = self.model(batch)
                predictions.append(batch_predictions.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def save_model(self, path: str = "transformer_model.pt"):
        """Save the model to disk."""
        torch.save(self.model.state_dict(), path)

    def plot_history(self):
        """Plot training and validation loss history."""
        fig, ax = plt.subplots()
        sns.lineplot(self.history["train_loss"], label="Train Loss", ax=ax)
        sns.lineplot(self.history["val_loss"], label="Validation Loss", ax=ax)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        fig.suptitle("Training and Validation Loss History")
        fig.legend()
        plt.tight_layout()
        plt.savefig("transformer_history.png", bbox_inches="tight")
        plt.show()


# --------------------------------
# Analysis and Visualization
# --------------------------------


class ModelAnalyzer:
    """Class for analyzing and visualizing model results."""

    def __init__(self, models: List[BaseModel], original_data: pd.DataFrame):
        """Initialize the model analyzer.

        Args:
            models: List of trained model objects
            original_data: Original DataFrame with behavior profiles
        """
        self.models = models
        self.original_data = original_data
        self.results = {}

    def compare_performance(self) -> pd.DataFrame:
        """Compare the performance of all models."""
        results = []

        for model in self.models:
            metrics = model.metrics.copy()
            metrics["model"] = model.name
            results.append(metrics)

        results_df = pd.DataFrame(results)
        # Reorder columns to put model name first
        cols = ["model"] + [col for col in results_df.columns if col != "model"]
        results_df = results_df[cols]

        self.results["performance"] = results_df
        return results_df

    def plot_performance_comparison(
        self,
        metrics: List[str] = [
            "mae",
            "rmse",
            "training_time",
            "memory_usage",
            "gpu_memory_usage",
        ],
    ):
        """Plot performance comparison across models.

        Args:
            metrics: List of metrics to compare
        """
        performance = self.results.get("performance")
        if performance is None:
            performance = self.compare_performance()

        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))

        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            sns.barplot(x="model", y=metric, data=performance, ax=axes[i])
            axes[i].set_title(f"Comparison of {metric}")
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("model_performance_comparison.png")
        plt.close()
        
    def categorize_behavior(self, pred_days):
        if pred_days < 0:
            return "early"
        elif pred_days <= 3:
            return "on_time"
        else:
            return "late"

    def analyze_predictions_by_behavior(
        self, X_test_orig, y_test, model_predictions: Dict[str, np.ndarray]
    ):
        """Analyze how predictions align with behavior profiles.

        Args:
            X_test_orig: Original test features (not preprocessed)
            y_test: True test values
            model_predictions: Dict mapping model names to their predictions
        """
        # Create a DataFrame for analysis
        analysis_df = X_test_orig.copy()
        analysis_df["true_days"] = y_test.values

        # Get client behavior profiles from original data
        # Only proceed if behavior_profile exists in the data
        if "behavior_profile" in self.original_data.columns:
            client_behaviors = self.original_data[
                ["client_encoded", "behavior_profile"]
            ].drop_duplicates()
            analysis_df = analysis_df.merge(
                client_behaviors, on="client_encoded", how="left"
            )

            # Add model predictions
            for model_name, preds in model_predictions.items():
                analysis_df[f"{model_name}_pred"] = preds
                analysis_df[f"{model_name}_error"] = np.abs(
                    analysis_df["true_days"] - preds
                )
                analysis_df[f"{model_name}_pred_type"] = analysis_df[f"{model_name}_pred"].apply(self.categorize_behavior)

            # Group by behavior profile and calculate mean error for each model
            behavior_results = []

            for behavior in analysis_df["behavior_profile"].unique():
                behavior_data = analysis_df[analysis_df["behavior_profile"] == behavior]

                result = {"behavior_profile": behavior}

                # Calculate mean error for each model
                for model_name in model_predictions.keys():
                    result[f"{model_name}_mean_error"] = behavior_data[
                        f"{model_name}_error"
                    ].mean()
                    result[f"{model_name}_pred_early"] = len(behavior_data[f"{model_name}_pred"][behavior_data[f"{model_name}_pred_type"] == "early"]) / len(behavior_data[f"{model_name}_pred"])
                    result[f"{model_name}_pred_on_time"] = len(behavior_data[f"{model_name}_pred"][behavior_data[f"{model_name}_pred_type"] == "on_time"]) / len(behavior_data[f"{model_name}_pred"])
                    result[f"{model_name}_pred_late"] = len(behavior_data[f"{model_name}_pred"][behavior_data[f"{model_name}_pred_type"] == "late"]) / len(behavior_data[f"{model_name}_pred"])

                behavior_results.append(result)

            behavior_df = pd.DataFrame(behavior_results)
            self.results["behavior_analysis"] = behavior_df
            return behavior_df
        else:
            print(
                "Warning: 'behavior_profile' column not found in data. Skipping behavior analysis."
            )
            return None

    def plot_behavior_analysis(self):
        """Plot analysis of predictions by behavior profile."""
        behavior_df = self.results.get("behavior_analysis")
        if behavior_df is None:
            print("Behavior analysis not available. Skipping behavior plot.")
            return

        # Reshape for plotting
        plot_df = behavior_df.melt(
            id_vars=["behavior_profile"],
            value_vars=[
                col for col in behavior_df.columns if col.endswith("_mean_error")
            ],
            var_name="model",
            value_name="mean_absolute_error",
        )

        # Clean up model names for display
        plot_df["model"] = plot_df["model"].str.replace("_mean_error", "")

        plt.figure(figsize=(12, 8))
        sns.barplot(
            x="behavior_profile", y="mean_absolute_error", hue="model", data=plot_df
        )
        plt.title("Model Error by Client Behavior Profile")
        plt.xlabel("Client Behavior Profile")
        plt.ylabel("Mean Absolute Error (days)")
        plt.xticks(rotation=45)
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig("model_error_by_behavior.png")
        plt.close()
        
        # Plot per model what it predicted per behavior
        # Reshape data for plotting proportions
        proportion_data = []
        for model_name in [col.replace("_pred_early", "") for col in behavior_df.columns if "_pred_early" in col]:
            for behavior in behavior_df["behavior_profile"]:
                proportion_data.append({
                    "behavior_profile": behavior,
                    "model": model_name,
                    "predicted_behavior": "early",
                    "proportion": behavior_df.loc[behavior_df["behavior_profile"] == behavior, f"{model_name}_pred_early"].values[0],
                })
                proportion_data.append({
                    "behavior_profile": behavior,
                    "model": model_name,
                    "predicted_behavior": "on_time",
                    "proportion": behavior_df.loc[behavior_df["behavior_profile"] == behavior, f"{model_name}_pred_on_time"].values[0],
                })
                proportion_data.append({
                    "behavior_profile": behavior,
                    "model": model_name,
                    "predicted_behavior": "late",
                    "proportion": behavior_df.loc[behavior_df["behavior_profile"] == behavior, f"{model_name}_pred_late"].values[0],
                })

        proportion_df = pd.DataFrame(proportion_data)

        # Create a figure with subplots for each model
        unique_models = proportion_df["model"].unique()
        num_models = len(unique_models)
        fig, axes = plt.subplots(num_models, 1, figsize=(14, 5 * num_models), sharex=True)

        if num_models == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot

        for ax, model_name in zip(axes, unique_models):
            model_data = proportion_df[proportion_df["model"] == model_name]
            sns.barplot(
            x="behavior_profile",
            y="proportion",
            hue="predicted_behavior",
            data=model_data,
            ci=None,
            ax=ax
            )
            ax.set_title(f"Proportions of Predicted Behavior for {model_name}")
            ax.set_xlabel("Client Behavior Profile")
            ax.set_ylabel("Proportion")
            ax.legend(title="Predicted Behavior")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("predicted_behavior_proportions_by_model.png")
        plt.close()

# --------------------------------
# Main Execution
# --------------------------------


def main():
    """Main execution function."""
    args = parse_args()
    batch_size = args.batch_size if args.batch_size else 256
    epochs = args.epochs if args.epochs else 1
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Step 1: Load and process data
    print("Loading and processing data...")
    processor = DataProcessor("toy_invoices_with_client_patterns.csv")
    data = processor.load_data()
    processor.prepare_features()
    y_train, y_test, X_train_orig, X_test_orig = processor.get_processed_data()

    # Step 2: Train models
    print("Training models...")
    models = [
        StatisticalPaymentPredictor(prediction_method="mean"),
        StatisticalPaymentPredictor(prediction_method="median"),
        StatisticalPaymentPredictor(prediction_method="mode"),
        StatisticalPaymentPredictor(prediction_method="weighted_mean"),
        StatisticalModel(),
        SimpleNeuralNetwork(),
        ComplexNeuralNetwork(),
        ClientPaymentPredictionModel(),
    ]

    print(data.head())

    for model in models:
        print(f"Training {model.name}...")
        try:
            model.train(X_train_orig, y_train, epochs=epochs, batch_size=batch_size)
        except:
            model.train(data)
        metrics = model.evaluate(X_test_orig, y_test)
        print(f"  Metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
        print(f"  Training time: {metrics['training_time']:.2f} seconds")
        print(f"  Memory usage: {metrics['memory_usage']:.2f} MB")
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            print(f"  GPU memory usage: {metrics['gpu_memory_usage']:.2f} MB")

    # Step 3: Analyze results
    print("\nAnalyzing results...")
    analyzer = ModelAnalyzer(models, data)
    performance_df = analyzer.compare_performance()
    print("\nPerformance comparison:")
    print(performance_df)
    analyzer.plot_performance_comparison()

    # Step 4: Analyze by behavior profile
    print("\nAnalyzing by behavior profile...")
    model_predictions = {model.name: model.predict(X_test_orig) for model in models}
    behavior_df = analyzer.analyze_predictions_by_behavior(
        X_test_orig, y_test, model_predictions
    )
    if behavior_df is not None:
        print("\nBehavior profile analysis:")
        print(behavior_df)
        analyzer.plot_behavior_analysis()

    print("\nAnalysis complete! Results saved as PNG files.")


if __name__ == "__main__":
    main()
