import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import imageio
import os
import pandas as pd
from matplotlib.lines import Line2D

# Set seaborn style
sns.set_style("whitegrid")


# Create finance-inspired dataset with nonlinear boundary
def generate_finance_dataset():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    df = pd.read_excel(data_url, header=1)

    # Select two numerical features for plotting
    X = df[["LIMIT_BAL", "PAY_0"]].values
    y = df["default payment next month"].values

    return X, y


# Define a simple neural net
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# Plot decision boundary
def plot_decision_boundary(model, X, y, epoch, filename, true_boundary_func=None):
    x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.4, cmap="RdBu")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k")

    # Build legend handles for class labels
    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="No Default",
            markerfacecolor=plt.cm.RdBu(0.2),
            markersize=8,
            markeredgecolor="k",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Default",
            markerfacecolor=plt.cm.RdBu(0.8),
            markersize=8,
            markeredgecolor="k",
        ),
    ]

    plt.title(f"Epoch {epoch}")
    plt.xlabel("Credit Limit")
    plt.ylabel("September payment delay (months)")
    if true_boundary_func:
        zz = true_boundary_func(xx, yy)
        contour = plt.contour(
            xx,
            yy,
            zz,
            levels=[0.5],
            colors="green",
            linewidths=1.5,
            linestyles="dashed",
        )
        boundary_handle = Line2D(
            [0],
            [0],
            color="green",
            linestyle="dashed",
            linewidth=1.5,
            label="True boundary",
        )
        plt.legend(
            handles=class_handles + [boundary_handle], title="Legend", loc="upper right"
        )
    else:
        plt.legend(handles=class_handles, title="Client Status", loc="upper right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Generate dataset
X, y = generate_finance_dataset()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert to torch tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

# Define model, loss, optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Function to generate the true decision boundary for comparison
def true_boundary(xx, yy):
    # Use the same nonlinear function as make_moons, adjusted to scaled space
    xx_orig = xx * scaler.scale_[0] + scaler.mean_[0]
    yy_orig = yy * scaler.scale_[1] + scaler.mean_[1]
    return np.sin(xx_orig * np.pi / 100) > yy_orig / 20


# Training loop with visualization
frames = []
os.makedirs("frames2", exist_ok=True)
for epoch in range(1, 101):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    filename = f"frames2/frame_{epoch:03d}.png"
    plot_decision_boundary(
        model, X_train, y_train, epoch, filename, true_boundary_func=true_boundary
    )
    frames.append(imageio.v2.imread(filename))

# Save GIF
imageio.mimsave("decision_boundary_evolution.gif", frames, duration=0.1)
print("GIF saved as 'decision_boundary_evolution.gif'")
