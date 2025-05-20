import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from tqdm import tqdm

sns.set_style("whitegrid")


# Generate meaningful financial data with comprehensive boundary
def generate_finance_data(n_samples=1000, noise=0.03):
    """
    Creates synthetic credit risk data with clear decision rules:
    High risk if:
    - Income < $50k AND Debt Ratio > 25%
    - Income $50k-$80k AND Debt Ratio > 35%
    - Income $80k-$150k AND Debt Ratio > 45%
    - Income > $150k AND Debt Ratio > 60%
    """
    np.random.seed(42)
    income = np.random.normal(110, 40, n_samples)  # More spread in incomes
    debt_ratio = np.random.normal(35, 12, n_samples)  # Centered around common ratios

    # Create clear decision boundary
    y = np.where(
        ((income < 50) & (debt_ratio > 25))
        | ((income >= 50) & (income < 80) & (debt_ratio > 35))
        | ((income >= 80) & (income < 150) & (debt_ratio > 45))
        | ((income >= 150) & (debt_ratio > 60)),
        1,
        0,
    )

    # Add minimal noise
    flip_mask = np.random.rand(n_samples) < noise
    y[flip_mask] = 1 - y[flip_mask]

    return np.column_stack([income, debt_ratio]), y


# Generate and split data
X, y = generate_finance_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors (same as before)
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)


# Neural network definition (same architecture)
class CreditRiskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


model = CreditRiskNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Create meshgrid covering full range
def create_meshgrid(x_range=(0, 200), y_range=(0, 100), step=1):
    xx, yy = np.meshgrid(
        np.arange(x_range[0], x_range[1], step), np.arange(y_range[0], y_range[1], step)
    )
    return xx, yy


xx, yy = create_meshgrid()
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])


# Enhanced true decision boundary
def true_decision_boundary(income):
    """Full coverage boundary for all income ranges"""
    boundary = np.zeros_like(income)
    boundary[income < 50] = 25
    boundary[(income >= 50) & (income < 80)] = 35
    boundary[(income >= 80) & (income < 150)] = 45
    boundary[income >= 150] = 60
    return boundary


# Training setup
epochs = 150
plot_interval = 1
images = []

for epoch in tqdm(range(epochs), desc="Training", total=epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % plot_interval == 0 or epoch == epochs - 1:
        model.eval()
        with torch.no_grad():
            Z = model(grid).reshape(xx.shape).numpy()

            plt.figure(figsize=(13, 8))
            # Plot predictions
            plt.contourf(
                xx, yy, Z, alpha=0.8, levels=[0, 0.5, 1], colors=["#AAAAFF", "#FFAAAA"]
            )

            # Plot true boundary
            income_range = np.linspace(0, 200, 200)
            debt_boundary = true_decision_boundary(income_range)
            plt.plot(
                income_range,
                debt_boundary,
                "k--",
                linewidth=2.5,
                label="True Risk Threshold",
            )

            # Plot data points
            plt.scatter(
                X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired, alpha=0.9
            )

            # Add class definitions
            plt.text(
                10,
                85,
                "High Risk Zone",
                fontsize=12,
                color="white",
                bbox=dict(facecolor="#FF4444", alpha=0.8, edgecolor="none"),
            )
            plt.text(
                130,
                10,
                "Low Risk Zone",
                fontsize=12,
                color="white",
                bbox=dict(facecolor="#4444FF", alpha=0.8, edgecolor="none"),
            )

            # Formatting
            plt.title(
                f"Epoch {epoch}\nCredit Risk Classification: Income vs. Debt Ratio",
                pad=20,
            )
            plt.xlabel("Annual Income ($k)", fontsize=12)
            plt.ylabel("Debt-to-Income Ratio (%)", fontsize=12)
            plt.xlim(0, 200)
            plt.ylim(0, 100)
            plt.legend(loc="upper right")

            # Save plot
            plt.savefig(
                f"epoch_{epoch}.png", bbox_inches="tight", dpi=150, transparent=False
            )
            plt.close()
            images.append(imageio.v2.imread(f"epoch_{epoch}.png"))

# Create GIF
imageio.mimsave(
    "credit_risk_classification.gif",
    images,
    duration=0.8,
    subrectangles=True,  # Enable optimization
    palettesize=256,  # Force consistent color palette
    fps=8,
)

print("Training complete! GIF saved as 'credit_risk_classification.gif'")
