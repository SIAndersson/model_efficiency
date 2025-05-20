import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import requests
from io import StringIO
import os
import argparse
import imageio
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import umap.umap_ as umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_X_y
from sklearn.utils import resample
from sklearn.utils import check_random_state
import optuna
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import roc_curve

sns.set_theme(style="whitegrid")


def plot_roc_curve(preds, target):
    no_skill_pred = [0 for _ in range(len(target))]
    ns_auc = roc_auc_score(target, no_skill_pred)

    model_auc = roc_auc_score(target, preds)

    print("No Skill: ROC AUC=%.3f" % (ns_auc))
    print("Model: ROC AUC=%.3f" % (model_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(target, no_skill_pred)
    lr_fpr, lr_tpr, _ = roc_curve(target, preds)

    # plot the roc curve for the model
    fig, ax = plt.subplots()
    plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
    plt.plot(lr_fpr, lr_tpr, marker=".", label="Model")
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # show the legend
    plt.legend()
    # show the plot
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
    plt.show()


class SupervisedPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = self.scaler.fit_transform(X)

        # Center class means
        class_means = np.array([X[y == label].mean(axis=0) for label in np.unique(y)])
        class_centered = X.copy()
        for label in np.unique(y):
            class_centered[y == label] -= class_means[label]

        self.pca.fit(class_centered)
        return self

    def transform(self, X):
        X = self.scaler.transform(X)
        return self.pca.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class KernelLDAWithNystrom(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=2,
        kernel="rbf",
        gamma=None,
        n_landmarks=1000,
        random_state=None,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.n_landmarks = n_landmarks
        self.random_state = random_state

    def _compute_kernel(self, X, Y=None):
        if self.kernel == "rbf":
            return rbf_kernel(X, Y, gamma=self.gamma)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def fit(self, X, y):
        rng = check_random_state(self.random_state)

        # Step 1: Select landmark points
        n_samples = X.shape[0]
        if self.n_landmarks > n_samples:
            raise ValueError("n_landmarks must be <= number of samples")

        landmark_indices = rng.choice(n_samples, self.n_landmarks, replace=False)
        self.X_landmarks_ = X[landmark_indices]

        # Step 2: Compute K_XL and K_LL
        K_XL = self._compute_kernel(X, self.X_landmarks_)  # (n x L)
        K_LL = self._compute_kernel(self.X_landmarks_)  # (L x L)

        # Step 3: Stabilize K_LL
        reg = 1e-6 * np.eye(K_LL.shape[0])
        K_LL += reg

        # Step 4: Compute Nyström projection Z = K_XL @ K_LL^{-1/2}
        sqrt_K_LL_inv = np.linalg.inv(np.linalg.cholesky(K_LL)).T  # K_LL^{-1/2}
        self.Z_ = K_XL @ sqrt_K_LL_inv

        # Step 5: Apply LDA only if valid
        self.classes_ = np.unique(y)
        if self.n_components <= min(self.Z_.shape[1], len(self.classes_) - 1):
            self.reducer_ = LinearDiscriminantAnalysis(n_components=self.n_components)
        else:
            # Fall back to PCA or identity
            if self.n_components > self.Z_.shape[1]:
                raise ValueError(
                    f"n_components={self.n_components} exceeds Nyström feature dim={self.Z_.shape[1]}"
                )
            self.reducer_ = PCA(n_components=self.n_components)

        self.Z_reduced_ = self.reducer_.fit_transform(self.Z_, y)

        return self

    def transform(self, X):
        # Step 1: Project new data using landmarks
        K_XL = self._compute_kernel(X, self.X_landmarks_)
        sqrt_K_LL_inv = np.linalg.inv(
            np.linalg.cholesky(
                self._compute_kernel(self.X_landmarks_)
                + 1e-6 * np.eye(self.n_landmarks)
            )
        ).T
        Z = K_XL @ sqrt_K_LL_inv

        return self.reducer_.transform(Z)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.Z_reduced_


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Neural Network Decision Boundary Visualization"
)
parser.add_argument(
    "--dim_reduction",
    type=str,
    default="pca",
    choices=["pca", "umap", "tsne", "spca", "iso", "klda"],
    help="Dimensionality reduction method: pca, tsne, spca, klda, iso, or umap",
)
parser.add_argument(
    "--num_epochs", type=int, default=30, help="Number of training epochs"
)
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--hidden_size", type=int, default=32, help="Size of hidden layer")
parser.add_argument(
    "--smoothing",
    type=float,
    default=2.0,
    help="Smoothing factor for decision boundary",
)
parser.add_argument(
    "--weight_decay", type=float, default=0.01, help="L2 regularization strength"
)
parser.add_argument("--parameter_tuning", action="store_true")
args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["figure.dpi"] = 100

# Download the Default of Credit Card Clients dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
print("Downloading dataset...")

# Use requests to download the file
response = requests.get(url)
if response.status_code != 200:
    print(f"Failed to download data: {response.status_code}")
    # Use a local backup or alternative approach
    # For this example, we'll use pandas to read directly from the URL
    df = pd.read_excel(url, header=1)
else:
    # Save the content to a temporary file and read with pandas
    with open("credit_card_default.xls", "wb") as f:
        f.write(response.content)
    df = pd.read_excel("credit_card_default.xls", header=1)
    # Clean up temporary file
    os.remove("credit_card_default.xls")

print("Dataset loaded successfully!")

# Cleaning of data
df = df.rename(columns={"PAY_0": "PAY_1"})
fil = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
df.loc[fil, "EDUCATION"] = 4

df.loc[df.MARRIAGE == 0, "MARRIAGE"] = 3

fil = (df.PAY_1 == -2) | (df.PAY_1 == -1) | (df.PAY_1 == 0)
df.loc[fil, "PAY_1"] = 0
fil = (df.PAY_2 == -2) | (df.PAY_2 == -1) | (df.PAY_2 == 0)
df.loc[fil, "PAY_2"] = 0
fil = (df.PAY_3 == -2) | (df.PAY_3 == -1) | (df.PAY_3 == 0)
df.loc[fil, "PAY_3"] = 0
fil = (df.PAY_4 == -2) | (df.PAY_4 == -1) | (df.PAY_4 == 0)
df.loc[fil, "PAY_4"] = 0
fil = (df.PAY_5 == -2) | (df.PAY_5 == -1) | (df.PAY_5 == 0)
df.loc[fil, "PAY_5"] = 0
fil = (df.PAY_6 == -2) | (df.PAY_6 == -1) | (df.PAY_6 == 0)
df.loc[fil, "PAY_6"] = 0

# Feature engineering
pay_columns = [f"PAY_AMT{i + 1}" for i in range(6)]
bill_columns = [f"BILL_AMT{i + 1}" for i in range(6)]
pay_status_cols = [f"PAY_{i + 1}" for i in range(6)]

for pay_col, bill_col in zip(pay_columns, bill_columns):
    df[f"{pay_col}_RATIO"] = df[pay_col] / (df[bill_col] + 1e-6)

df["PAY_RATIO_AVG"] = df[[f"{col}_RATIO" for col in pay_columns]].mean(axis=1)
df["PAY_RATIO_STD"] = df[[f"{col}_RATIO" for col in pay_columns]].std(axis=1)
df["PAY_RATIO_MIN"] = df[[f"{col}_RATIO" for col in pay_columns]].min(axis=1)

df["UTIL_RATE1"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
df["UTIL_AVG"] = df[bill_columns].mean(axis=1) / df["LIMIT_BAL"]
df["UTIL_MAX"] = df[bill_columns].max(axis=1) / df["LIMIT_BAL"]

# Delinquency counts & flags
df["NUM_LATE"] = (df[pay_status_cols] > 0).sum(axis=1)
df["EVER_60_PLUS"] = (df[pay_status_cols] > 1).any(axis=1).astype(int)

# 3. Temporal & Trend Features
# Month-to-month differences
for i in range(1, 6):
    df[f"BILL_DIFF_{i}_{i + 1}"] = df[f"BILL_AMT{i}"] - df[f"BILL_AMT{i + 1}"]
    df[f"PAY_DIFF_{i}_{i + 1}"] = df[f"PAY_AMT{i}"] - df[f"PAY_AMT{i + 1}"]

# Rolling window stats (3-month)
df["rolling_bill_mean_3"] = df[bill_columns[:3]].mean(axis=1)
df["rolling_pay_std_3"] = df[pay_columns[:3]].std(axis=1)


# Prepare the data
X = df.drop(["ID", "default payment next month"], axis=1).values
y = df["default payment next month"].values

# Split into training and testing sets FIRST
X_train_original, X_test_original, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_original)
X_test_scaled = scaler.transform(X_test_original)

# Apply dimensionality reduction to reduce to 2 dimensions for visualization
print(f"Applying {args.dim_reduction.upper()} for dimensionality reduction...")
if args.dim_reduction == "pca":
    reducer = PCA(n_components=2)
    X_reduced_train = reducer.fit_transform(X_train_scaled)
    X_reduced_test = reducer.transform(X_test_scaled)
    explained_variance = reducer.explained_variance_ratio_
    print(f"Variance explained: {sum(explained_variance):.2%}")
elif args.dim_reduction == "tsne":
    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    X_reduced_train = reducer.fit_transform(X_train_scaled)
    X_reduced_test = reducer.fit_transform(
        X_test_scaled
    )  # Note: t-SNE has no transform
elif args.dim_reduction == "spca":
    reducer = SupervisedPCA(n_components=2)
    X_reduced_train = reducer.fit_transform(X_train_scaled, y_train)
    X_reduced_test = reducer.transform(X_test_scaled)
elif args.dim_reduction == "klda":
    reducer = KernelLDAWithNystrom(n_components=2, gamma=0.1)
    X_reduced_train = reducer.fit_transform(X_train_scaled, y_train)
    X_reduced_test = reducer.transform(X_test_scaled)
elif args.dim_reduction == "iso":
    reducer = Isomap(
        n_components=2,
        n_jobs=-1,
    )
    X_reduced_train = reducer.fit_transform(X_train_scaled)
    X_reduced_test = reducer.transform(X_test_scaled)
else:  # umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_reduced_train = reducer.fit_transform(X_train_scaled)
    X_reduced_test = reducer.transform(X_test_scaled)

# Create PyTorch tensors from the original scaled data
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# Define an improved neural network with more regularization
class ImprovedNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=32):
        super(ImprovedNN, self).__init__()
        # Using larger hidden sizes and adding batch normalization
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.layer3 = nn.Linear(hidden_size, 2)  # Binary classification

        # Using softer activation function
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Apply batch normalization for stability and smoother gradients
        x = self.activation(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return x


class EnhancedNN(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(0.4),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.out = nn.Linear(hidden_size // 2, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.out(x)


class Swish(nn.Module):
    """Swish activation function (x * sigmoid(x))"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """Modular residual block with configurable normalization and stochastic depth"""

    def __init__(
        self,
        input_dim,
        output_dim,
        dropout_rate=0.1,
        use_bn=True,
        use_stochastic_depth=False,
        stochastic_depth_prob=0.5,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(output_dim) if use_bn else nn.Identity()
        self.activation = Swish()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_prob = stochastic_depth_prob

        # Skip connection with dimension matching
        self.skip = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)

        if self.use_stochastic_depth and self.training:
            if torch.rand(1) < self.stochastic_depth_prob:
                return residual

        out = self.dense(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out + residual


class CreditResNet(nn.Module):
    """Modular ResNet architecture for credit default prediction"""

    def __init__(
        self,
        input_dim,
        hidden_dims=[1024, 64, 16],
        dropout_rates=[0.1, 0.1, 0.1],
        use_bn_flags=[False, True, True],
        use_stochastic_depth=False,
        stochastic_depth_probs=[0.5, 0.5, 0.5],
        output_dim=2,
    ):
        super().__init__()

        # Initial batch normalization
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Create residual blocks
        self.blocks = nn.ModuleList()
        prev_dim = input_dim
        for i, (h_dim, drop_rate, use_bn, sd_prob) in enumerate(
            zip(hidden_dims, dropout_rates, use_bn_flags, stochastic_depth_probs)
        ):
            self.blocks.append(
                ResidualBlock(
                    input_dim=prev_dim,
                    output_dim=h_dim,
                    dropout_rate=drop_rate,
                    use_bn=use_bn,
                    use_stochastic_depth=use_stochastic_depth,
                    stochastic_depth_prob=sd_prob,
                )
            )
            prev_dim = h_dim

        # Output layer
        self.output = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.input_bn(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


# Initialize model, loss function, and optimizer with weight decay (L2 regularization)
# model = ImprovedNN(input_size=X_train_scaled.shape[1], hidden_size=args.hidden_size)
model = EnhancedNN(input_size=X_train_scaled.shape[1], hidden_size=args.hidden_size)
"""model = CreditResNet(
    input_dim=X_train_scaled.shape[1],  # Number of features in your dataset
    hidden_dims=[1024, 64, 16],
    dropout_rates=[0.1, 0.1, 0.1],
    use_bn_flags=[False, True, True],  # Match original architecture
    use_stochastic_depth=True,
    stochastic_depth_probs=[0.2, 0.2, 0.2],
)"""

# Calculate class weights
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts  # Inverse of class frequency
class_weights = torch.FloatTensor(class_weights / class_weights.sum())

# Modify the loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,  # Add L2 regularization for smoother boundaries
)
# optimizer = torch.optim.NAdam(model.parameters(), lr=0.01, weight_decay=4e-4)


# Create a mesh grid for plotting the decision boundary
def plot_decision_boundary(
    model, reducer, X_reduced, y, epoch, ax=None, smoothing_sigma=2.0
):
    # Set min and max values with some padding
    x_min, x_max = X_reduced[:, 0].min() - 0.5, X_reduced[:, 0].max() + 0.5
    y_min, y_max = X_reduced[:, 1].min() - 0.5, X_reduced[:, 1].max() + 0.5

    # Create a mesh grid with appropriate step size (larger h for lower resolution)
    h = 0.15  # Increased from 0.1 to reduce noise
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Convert to PyTorch tensors
    grid_reduced = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    # Inverse transform to original feature space
    try:
        try:
            grid_original = (
                reducer.inverse_transform(grid_reduced).numpy().astype(np.float32)
            )
        except AttributeError:
            grid_original = reducer.inverse_transform(grid_reduced).astype(np.float32)
    except AttributeError:
        raise ValueError(f"{args.dim_reduction} does not support inverse_transform")

    grid_tensor = torch.FloatTensor(grid_original)

    # Get predictions
    with torch.no_grad():
        outputs = model(grid_tensor)
        probs = torch.softmax(outputs, dim=1)[
            :, 1
        ].numpy()  # Get probability of class 1

    # Reshape back to mesh grid shape
    Z = probs.reshape(xx.shape)

    # Apply Gaussian smoothing to the decision boundary
    Z_smoothed = gaussian_filter(Z, sigma=smoothing_sigma)

    # Convert probabilities to binary predictions with threshold 0.5
    predicted = (Z_smoothed >= 0.5).astype(int)

    # Plot the decision boundary
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        ax.clear()

    # Create custom colormap
    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#0000FF"])

    # Plot the decision boundary
    ax.contourf(xx, yy, predicted, alpha=0.4, cmap=cmap_light)

    # Add contour lines for probability levels (shows smooth transitions)
    contour = ax.contour(
        xx,
        yy,
        Z_smoothed,
        levels=[0.3, 0.5, 0.7],
        colors=["black"],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        linewidths=[1, 2, 1],
    )
    ax.clabel(contour, inline=True, fontsize=8)

    # Plot the data points
    scatter = ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=y,
        cmap=cmap_bold,
        edgecolors="k",
        s=40,
        alpha=0.7,
    )

    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    legend1.get_texts()[0].set_text("Not Default")
    legend1.get_texts()[1].set_text("Default")
    ax.add_artist(legend1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Decision Boundary (Epoch {epoch})")

    return ax


# Function to generate an approximate "true" decision boundary
def generate_true_boundary(X, y, dim_reduction_method):
    from sklearn.svm import SVC

    # Train an SVM with a non-linear kernel as a proxy for the "true" boundary
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
    svm.fit(X, y)

    # Set min and max values with some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create a mesh grid
    h = 0.05  # Step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Get probability predictions
    Z_probs = svm.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_probs = Z_probs.reshape(xx.shape)

    # Apply Gaussian filter for smoothing
    Z_probs_smooth = gaussian_filter(Z_probs, sigma=2.0)
    Z = (Z_probs_smooth >= 0.5).astype(int)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create custom colormap
    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#0000FF"])

    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)

    # Add contour lines for probability levels
    contour = ax.contour(
        xx,
        yy,
        Z_probs_smooth,
        levels=[0.3, 0.5, 0.7],
        colors=["black"],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        linewidths=[1, 2, 1],
    )
    ax.clabel(contour, inline=True, fontsize=8)

    # Plot the data points
    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors="k", s=40, alpha=0.7
    )

    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    legend1.get_texts()[0].set_text("Not Default")
    legend1.get_texts()[1].set_text("Default")
    ax.add_artist(legend1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(
        f'Approximate "True" Decision Boundary using {dim_reduction_method.upper()} (SVM RBF Kernel)'
    )

    filename = f"true_decision_boundary_{dim_reduction_method}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return filename


# Plot the true decision boundary
print(f"Generating approximate 'true' decision boundary using {args.dim_reduction}...")
# true_boundary_file = generate_true_boundary(X_train, y_train, args.dim_reduction)

# Training the model and capturing boundary evolution
print("Training neural network and capturing decision boundary evolution...")
num_epochs = args.num_epochs
frames = []

# Set up figure for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

# Store the training loss history
train_loss_history = []
val_loss_history = []
val_f1_history = []

# Learning rate scheduler for better convergence
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)


def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer_cat = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
    w_decay = trial.suggest_float("weight_decay", 0.00001, 0.01, log=True)
    scheduler_cat = trial.suggest_categorical("scheduler", ["Cosine", "Plateau"])

    model = EnhancedNN(input_size=X_train_scaled.shape[1], hidden_size=hidden_size)
    if optimizer_cat == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=w_decay)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=lr, nesterov=True, weight_decay=w_decay, momentum=0.9
        )

    if scheduler_cat == "Cosine":
        lrscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    elif scheduler_cat == "Plateau":
        lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    else:
        lrscheduler = None

    for epoch in tqdm(range(30), desc="Training model...", total=30):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)

        if scheduler_cat == "Plateau":
            lrscheduler.step(epoch_loss)
        elif scheduler_cat == "Cosine":
            lrscheduler.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            validation_accuracy = (predicted == y_test_tensor).sum().item() / len(
                y_test_tensor
            )

            probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
            predicted = (probs > 0.5).astype(int)

            val_f1 = f1_score(y_test_tensor.numpy(), predicted)

        # trial.report(val_f1, epoch)

        # if trial.should_prune():
        # raise optuna.exceptions.TrialPruned()

    return val_f1


if args.parameter_tuning:
    study = optuna.create_study(
        direction="maximize",
        # pruner=optuna.pruners.HyperbandPruner(min_resource=2, reduction_factor=3),
    )
    study.optimize(objective, n_trials=250)
    print(f"Best params is {study.best_params} with value {study.best_value}")
    exit()


# ax.clear()
# plot_decision_boundary(model, reducer, X_reduced_train, y_train, 0, ax, args.smoothing)
# Save the plot as a frame
# fig.canvas.draw()
# frame = np.array(fig.canvas.renderer.buffer_rgba())
# frames.append(frame)

# Training loop
for epoch in tqdm(range(num_epochs), desc="Training model...", total=num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_loss_history.append(epoch_loss)

    # Adjust learning rate based on loss
    scheduler.step(epoch_loss)

    # Print statistics
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Switch to evaluation mode for visualization
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        loss = criterion(outputs, y_test_tensor)
        val_loss = loss.item()

        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
        predicted = (probs > 0.5).astype(int)
        val_f1 = f1_score(y_test_tensor.numpy(), predicted)
    val_loss_history.append(val_loss)
    val_f1_history.append(val_f1)

    # Plot decision boundary every epoch
    ax.clear()
    plot_decision_boundary(
        model, reducer, X_reduced_train, y_train, epoch + 1, ax, args.smoothing
    )

    # Save the plot as a frame
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)

plt.close()

# Save frames as GIF
print("Creating GIF animation...")

# Save the animation with improved framerate
output_gif = f"smooth_decision_boundary_{args.dim_reduction}_full.gif"
imageio.mimsave(output_gif, frames, fps=3)

print(f"Training completed and visualization saved as '{output_gif}'")

# Plot the loss curve
fig, ax1 = plt.subplots()
ln1 = ax1.plot(range(1, num_epochs + 1), train_loss_history, marker="o", label="Train")
ln2 = ax1.plot(
    range(1, num_epochs + 1), val_loss_history, marker="x", label="Validation"
)
plt.suptitle("Training Loss")
plt.xlabel("Epoch")
ax1.set_ylabel("Loss")

ax2 = ax1.twinx()
ln3 = ax2.plot(
    range(1, num_epochs + 1),
    val_f1_history,
    marker="*",
    label="F1-Score",
    color="green",
)
ax2.set_ylabel("F1-score")
ax1.grid(True)

lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs)

loss_curve_file = f"training_loss_{args.dim_reduction}_full.png"
fig.tight_layout()
plt.savefig(loss_curve_file, dpi=300, bbox_inches="tight")
plt.close()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"Test Accuracy: {accuracy:.4f}")

    class_0_idx = torch.argwhere(y_test_tensor == 0)
    class_1_idx = torch.argwhere(y_test_tensor == 1)

    class_0_acc = (
        predicted[class_0_idx] == y_test_tensor[class_0_idx]
    ).sum().item() / len(predicted[class_0_idx])
    class_1_acc = (
        predicted[class_1_idx] == y_test_tensor[class_1_idx]
    ).sum().item() / len(predicted[class_1_idx])

    print(f"Class 0 (not default) accuracy: {class_0_acc}")
    print(f"Class 1 (default) accuracy: {class_1_acc}")

    probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
    predicted = (probs > 0.5).astype(int)

    plot_roc_curve(probs, y_test_tensor.numpy())
    print(f"Test AUC: {roc_auc_score(y_test_tensor.numpy(), probs):.4f}")
    print(f"F1-Score: {f1_score(y_test_tensor.numpy(), predicted):.4f}")

# Final model's decision boundary
plt.figure(figsize=(10, 8))
ax = plt.gca()
plot_decision_boundary(
    model, reducer, X_reduced_test, y_test, "Final", ax, args.smoothing
)
final_boundary_file = f"final_smooth_boundary_{args.dim_reduction}_full.png"
plt.savefig(final_boundary_file, dpi=300, bbox_inches="tight")
plt.close()

print("Script completed successfully!")
print("Output files:")
print(f"1. {output_gif} - Animation of smooth decision boundary evolution")
# print(f"2. {true_boundary_file} - Approximate 'true' decision boundary using SVM")
print(f"3. {final_boundary_file} - Final neural network decision boundary")
print(f"4. {loss_curve_file} - Training loss curve")
print("\nCommand line options:")
print("--dim_reduction [pca|tsne] - Choose dimensionality reduction method")
print("--num_epochs [int] - Number of training epochs")
print("--learning_rate [float] - Learning rate for optimization")
print("--hidden_size [int] - Hidden layer size")
print("--smoothing [float] - Smoothing factor for decision boundary (default: 2.0)")
print("--weight_decay [float] - L2 regularization strength (default: 0.01)")
print("\nExample usage:")
print(
    "python smoothed_decision_boundary.py --dim_reduction tsne --num_epochs 50 --learning_rate 0.001 --hidden_size 32 --smoothing 3.0 --weight_decay 0.02"
)
