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


# Generate meaningful financial data with non-linear decision boundary
def generate_finance_data(n_samples=1000, noise=0.1):
    """
    Creates synthetic credit risk data with meaningful non-linear decision boundary.
    Decision rule:
    - High risk if (debt_ratio > 35% AND income < $80k) OR
                  (debt_ratio > 50% AND income < $120k) OR
                  (debt_ratio > 25% AND income < $50k)
    - With some added noise
    """
    np.random.seed(42)
    income = np.random.normal(100, 30, n_samples)  # Annual income in $k
    debt_ratio = np.random.normal(30, 10, n_samples)  # Debt-to-income ratio in %

    # Create meaningful non-linear decision boundary
    y = np.where(
        ((debt_ratio > 35) & (income < 80))
        | ((debt_ratio > 50) & (income < 120))
        | ((debt_ratio > 25) & (income < 50)),
        1,
        0,
    )

    # Add noise
    flip_mask = np.random.rand(n_samples) < noise
    y[flip_mask] = 1 - y[flip_mask]

    return np.column_stack([income, debt_ratio]), y


# Generate and split data
X, y = generate_finance_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)


# Define neural network
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
optimizer = optim.Adam(model.parameters(), lr=0.005)


# Create meshgrid for visualization
def create_meshgrid(x_range=(0, 200), y_range=(0, 80), step=1):
    xx, yy = np.meshgrid(
        np.arange(x_range[0], x_range[1], step), np.arange(y_range[0], y_range[1], step)
    )
    return xx, yy


xx, yy = create_meshgrid()
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])


# True decision boundary calculation
def true_decision_boundary(income):
    """Calculate the true debt ratio thresholds for different income levels"""
    boundary = np.zeros_like(income)
    boundary[income < 50] = 25
    boundary[(income >= 50) & (income < 80)] = 35
    boundary[(income >= 80) & (income < 120)] = 50
    boundary[income >= 120] = np.inf  # No upper boundary for high incomes
    return boundary


# Training setup
epochs = 150
plot_interval = 1
images = []

# Training loop with improved visualization
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

            plt.figure(figsize=(12, 7))
            # Plot neural network predictions
            plt.contourf(
                xx, yy, Z, alpha=0.3, levels=[0, 0.5, 1], colors=["#FFAAAA", "#AAAAFF"]
            )

            # Plot true decision boundary
            income_range = np.linspace(0, 200, 200)
            debt_boundary = true_decision_boundary(income_range)
            plt.plot(
                income_range,
                debt_boundary,
                "k--",
                linewidth=2,
                label="True Decision Boundary",
            )

            # Plot data points
            scatter = plt.scatter(
                X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired, alpha=0.6
            )

            plt.title(
                f"Epoch {epoch}\nCredit Risk Decision Boundary Evolution\nIncome vs. Debt-to-Income Ratio"
            )
            plt.xlabel("Annual Income ($k)")
            plt.ylabel("Debt-to-Income Ratio (%)")
            plt.xlim(0, 200)
            plt.ylim(0, 80)
            plt.legend()

            # Save plot
            plt.savefig(f"epoch_{epoch}.png", bbox_inches="tight")
            plt.close()
            images.append(imageio.imread(f"epoch_{epoch}.png"))

# Create GIF
imageio.mimsave("credit_risk_decision_boundary.gif", images, duration=0.8)

print("Training complete! GIF saved as 'credit_risk_decision_boundary.gif'")
