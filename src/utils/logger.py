import csv
import os
from datetime import datetime

LOG_PATH = "experiments/experiment_log.csv"


def log_experiment(config, mean_acc, std_acc):

    os.makedirs("experiments", exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": config.get("model"),
        "pooling": config.get("pooling"),
        "hidden_dim": config.get("hidden_dim"),
        "heads": config.get("heads"),
        "lr": config.get("lr"),
        "batch_size": config.get("batch_size"),
        "epochs": config.get("epochs"),
        "scheduler": config.get("scheduler"),
        "accuracy_mean": round(mean_acc, 4),
        "accuracy_std": round(std_acc, 4)
    }

    file_exists = os.path.isfile(LOG_PATH)

    with open(LOG_PATH, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"\n📊 Experiment logged to {LOG_PATH}")