import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style="whitegrid", context="talk")

# ------------------------
# Configuration parameters
# ------------------------
NUM_CLIENTS = 5000
INVOICES_PER_CLIENT = 500
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2025, 1, 1)
SEED = 42

# Payment behavior settings
CLIENT_BEHAVIOR_PROFILES = {
    "always_early": {"early": 0.7, "on_time": 0.2, "late": 0.1},
    "mostly_on_time": {"early": 0.1, "on_time": 0.8, "late": 0.1},
    "often_late": {"early": 0.1, "on_time": 0.3, "late": 0.6},
    "random_behavior": {"early": 0.3, "on_time": 0.4, "late": 0.3},
}

# Delay distributions
LATE_MEAN_DAYS = 10
LATE_STD_DAYS = 5
EARLY_MEAN_DAYS = 3
EARLY_STD_DAYS = 2

# ------------------------
# Dataset generation logic
# ------------------------


def generate_clients(
    num_clients: int, seed: int = 42
) -> Tuple[Dict[str, Dict[str, float]], List]:
    random.seed(seed)
    clients = {}
    behavior_profiles = list(CLIENT_BEHAVIOR_PROFILES.keys())
    profile_list = []
    for i in range(num_clients):
        client_name = f"Client_{i + 1}"
        behavior = random.choice(behavior_profiles)
        clients[client_name] = CLIENT_BEHAVIOR_PROFILES[behavior]
        profile_list.append(behavior)
    return clients, profile_list


def random_date(start: datetime, end: datetime) -> datetime:
    return start + timedelta(days=random.randint(0, (end - start).days))


def generate_invoice_data(
    client: str,
    behavior_profile: Dict[str, float],
    num_invoices: int,
    behavior_type: str,
    START_DATE=START_DATE,
    END_DATE=END_DATE,
) -> pd.DataFrame:
    invoices = []
    for _ in range(num_invoices):
        due_date = random_date(START_DATE, END_DATE)

        # Seasonal adjustment
        month = due_date.month
        weekday = due_date.weekday()

        seasonal_adjustment = 0
        if month in [7, 8, 12]:  # Summer and Christmas
            seasonal_adjustment += np.random.normal(2, 1)  # Small delay increase
        if weekday in [0, 1]:  # Monday and Tuesday
            seasonal_adjustment += np.random.normal(1, 0.5)  # Slight delay increase

        # Determine payment behavior for this invoice based on client's profile
        behavior = random.choices(
            ["early", "on_time", "late"],
            weights=[
                behavior_profile["early"],
                behavior_profile["on_time"],
                behavior_profile["late"],
            ],
            k=1,
        )[0]

        if behavior == "early":
            days_diff = int(
                np.clip(np.random.normal(EARLY_MEAN_DAYS, EARLY_STD_DAYS), 1, 30)
            )
            payment_date = due_date - timedelta(days=days_diff)
        elif behavior == "on_time":
            payment_date = due_date
        else:  # late
            days_diff = int(
                np.clip(
                    np.random.normal(
                        LATE_MEAN_DAYS + seasonal_adjustment, LATE_STD_DAYS
                    ),
                    1,
                    60,
                )
            )
            payment_date = due_date + timedelta(days=days_diff)

        invoices.append(
            {
                "client": client,
                "due_date": due_date,
                "payment_date": payment_date,
                "days_to_payment": (payment_date - due_date).days,
                "payment_behavior": behavior,
                "behavior_profile": behavior_type,
            }
        )

    return pd.DataFrame(invoices)


def generate_full_dataset(
    num_clients: int, invoices_per_client: int, seed: int = 42
) -> tuple:
    random.seed(seed)
    np.random.seed(seed)

    client_profiles, behaviors = generate_clients(num_clients, seed)
    all_invoices = pd.concat(
        [
            generate_invoice_data(client, profile, invoices_per_client, behaviors[i])
            for i, (client, profile) in tqdm(
                enumerate(client_profiles.items()),
                desc="Generating invoices",
                total=num_clients,
            )
        ],
        ignore_index=True,
    )

    future_invoices = pd.concat(
        [
            generate_invoice_data(
                client,
                profile,
                200,
                behaviors[i],
                START_DATE=datetime(2025, 2, 1),
                END_DATE=datetime(2026, 1, 1),
            )
            for i, (client, profile) in tqdm(
                enumerate(client_profiles.items()),
                desc="Future invoices...",
                total=num_clients,
            )
        ],
        ignore_index=True,
    )

    return all_invoices, future_invoices


# ------------------------
# Save / Run
# ------------------------

if __name__ == "__main__":
    dataset, future_dataset = generate_full_dataset(
        NUM_CLIENTS, INVOICES_PER_CLIENT, SEED
    )
    print(dataset.head())
    # Print distribution of payment behavior
    print(dataset["payment_behavior"].value_counts(normalize=True))
    # Print distribution of payment delays
    print(dataset["days_to_payment"].describe())

    # Plot the distribution of days_to_payment using seaborn

    plt.figure(figsize=(10, 6))
    sns.histplot(dataset["days_to_payment"], bins=30, kde=True, alpha=0.7)
    plt.title("Distribution of Days to Payment")
    plt.xlabel("Days to Payment")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("days_to_payment_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    dataset.to_csv("toy_invoices_with_client_patterns.csv", index=False)
    future_dataset.to_csv("toy_invoices_future.csv", index=False)
