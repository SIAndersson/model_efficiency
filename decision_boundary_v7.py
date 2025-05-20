import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Neural Network Decision Boundary Visualization"
)
parser.add_argument(
    "--dim_reduction",
    type=str,
    default="pca",
    choices=["pca", "tsne"],
    help="Dimensionality reduction method: pca or tsne",
)
parser.add_argument(
    "--num_epochs", type=int, default=30, help="Number of training epochs"
)
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--hidden_size", type=int, default=16, help="Size of hidden layer")
args = parser.parse_args()

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

# Prepare the data
X = df.drop(["ID", "default payment next month"], axis=1).values
y = df["default payment next month"].values

# Apply dimensionality reduction to reduce to 2 dimensions for visualization
print(f"Applying {args.dim_reduction.upper()} for dimensionality reduction...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if args.dim_reduction == "pca":
    reducer = PCA(n_components=2)
    X_reduced = reducer.fit_transform(X_scaled)
    explained_variance = reducer.explained_variance_ratio_
    print(
        f"Variance explained by the first two principal components: {sum(explained_variance):.2%}"
    )
else:  # t-SNE
    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    X_reduced = reducer.fit_transform(X_scaled)
    print(f"t-SNE dimensionality reduction completed")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

# Create PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=16):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, 2)  # Binary classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# Initialize model, loss function, and optimizer
model = SimpleNN(hidden_size=args.hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


# Create a mesh grid for plotting the decision boundary
def plot_decision_boundary(model, X, y, epoch, ax=None):
    # Set min and max values with some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create a mesh grid
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Convert to PyTorch tensors
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    # Get predictions
    with torch.no_grad():
        outputs = model(grid)
        _, predicted = torch.max(outputs, 1)

    # Reshape back to mesh grid shape
    predicted = predicted.numpy().reshape(xx.shape)

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
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Decision Boundary (Epoch {epoch})")

    return ax


# Function to generate an approximate "true" decision boundary
def generate_true_boundary(X, y, dim_reduction_method):
    from sklearn.svm import SVC

    # Train an SVM with a non-linear kernel as a proxy for the "true" boundary
    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X, y)

    # Set min and max values with some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create a mesh grid
    h = 0.05  # Step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Get predictions
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create custom colormap
    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#0000FF"])

    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)

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

# Training loop
for epoch in tqdm(range(num_epochs), desc="Training model...", total=num_epochs):
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

    # Print statistics
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Plot decision boundary every epoch
    ax.clear()
    plot_decision_boundary(model, X_train, y_train, epoch + 1, ax)

    # Save the plot as a frame
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)

plt.close()

# Save frames as GIF
print("Creating GIF animation...")

# Save the animation with improved framerate
output_gif = f"decision_boundary_evolution_{args.dim_reduction}.gif"
imageio.mimsave(output_gif, frames, fps=3)

print(f"Training completed and visualization saved as '{output_gif}'")

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_loss_history, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
loss_curve_file = f"training_loss_{args.dim_reduction}.png"
plt.savefig(loss_curve_file, dpi=300, bbox_inches="tight")
plt.close()

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"Test Accuracy: {accuracy:.4f}")

# Final model's decision boundary
plt.figure(figsize=(10, 8))
ax = plt.gca()
plot_decision_boundary(model, X_test, y_test, "Final", ax)
final_boundary_file = f"final_decision_boundary_{args.dim_reduction}.png"
plt.savefig(final_boundary_file, dpi=300, bbox_inches="tight")
plt.close()

print("Script completed successfully!")
print("Output files:")
print(f"1. {output_gif} - Animation of decision boundary evolution")
# print(f"2. {true_boundary_file} - Approximate 'true' decision boundary using SVM")
print(f"3. {final_boundary_file} - Final neural network decision boundary")
print(f"4. {loss_curve_file} - Training loss curve")
print("\nCommand line options:")
print("--dim_reduction [pca|tsne] - Choose dimensionality reduction method")
print("--num_epochs [int] - Number of training epochs")
print("--learning_rate [float] - Learning rate for optimization")
print("--hidden_size [int] - Hidden layer size")
print("\nExample usage:")
print(
    "python script.py --dim_reduction tsne --num_epochs 50 --learning_rate 0.0005 --hidden_size 32"
)
