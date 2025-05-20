import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import imageio
import os

# Set the style to white grid (seaborn)
sns.set_style("whitegrid")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Generate finance-related dataset with non-linear decision boundary
def generate_finance_data(n_samples=1000):
    """
    Generate a synthetic finance dataset with a non-linear decision boundary.
    X1: Price-to-Earnings Ratio (P/E)
    X2: Debt-to-Equity Ratio
    Y: Investment Decision (Buy: 1, Sell: 0)
    """
    # Create features
    x1 = np.random.uniform(5, 30, n_samples)  # P/E ratio (typical range)
    x2 = np.random.uniform(0, 2.5, n_samples)  # Debt-to-Equity ratio

    # Create a non-linear decision boundary: Companies with either:
    # 1. Good P/E (< 15) and reasonable debt (< 1.2)
    # 2. Or those with higher P/E but very low debt
    y = np.zeros(n_samples)

    # Decision rule 1: Good P/E and reasonable debt
    mask1 = (x1 < 15) & (x2 < 1.2)

    # Decision rule 2: Higher P/E but very low debt
    mask2 = (x1 >= 15) & (x1 < 25) & (x2 < 0.5)

    # Decision rule 3: Any company with extremely high P/E is not good regardless of debt
    mask3 = x1 >= 25

    # Apply the rules
    y[mask1 | mask2] = 1
    y[mask3] = 0

    # Add some noise to make it more realistic
    noise_indices = np.random.choice(
        n_samples, size=int(0.1 * n_samples), replace=False
    )
    y[noise_indices] = 1 - y[noise_indices]

    # Combine into a dataset
    X = np.column_stack((x1, x2))

    return X, y


# Generate the data
X, y = generate_finance_data(1000)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)


# Define the neural network
class FinanceNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1):
        super(FinanceNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


# Initialize the model
model = FinanceNN()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create a directory for frames
os.makedirs("frames", exist_ok=True)


# Function to plot decision boundary
def plot_decision_boundary(model, X, y, epoch, filename):
    # Set figure size and create the plot
    plt.figure(figsize=(10, 8))

    # Define the mesh grid for plotting
    h = 0.1  # Step size
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

    # Predict on the mesh grid
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model(torch.FloatTensor(grid_points)).detach().numpy()
    Z = Z.reshape(xx1.shape)

    # Create color maps
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00"])

    # Plot the decision boundary
    plt.contourf(xx1, xx2, Z, alpha=0.6, cmap=cmap_light)

    # Plot the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=50)

    plt.title(f"Epoch {epoch}: Neural Network Decision Boundary", fontsize=14)
    plt.xlabel("Price-to-Earnings Ratio (P/E)", fontsize=12)
    plt.ylabel("Debt-to-Equity Ratio", fontsize=12)

    # Add a legend
    legend_labels = {0: "Sell", 1: "Buy"}
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=legend_labels[i],
            markerfacecolor=c,
            markersize=10,
        )
        for i, c in zip([0, 1], ["#FF0000", "#00FF00"])
    ]
    plt.legend(handles=legend_elements, title="Investment Decision", fontsize=10)

    # Save the figure
    plt.savefig(filename)
    plt.close()


# Training parameters
epochs = 125
save_every = 1  # Save visualization every N epochs
frames = []

# Training loop
for epoch in range(epochs + 1):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Print every 5 epochs
    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

    # Save visualization
    if epoch % save_every == 0 or epoch == epochs:
        filename = f"frames/epoch_{epoch:03d}.png"
        plot_decision_boundary(model, X, y, epoch, filename)
        frames.append(filename)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Create the GIF
print("Creating GIF animation...")
with imageio.get_writer(
    "neural_network_learning.gif", mode="I", duration=0.3
) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

print("GIF saved as 'neural_network_learning.gif'")
