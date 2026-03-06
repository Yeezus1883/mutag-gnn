import json
import os
from datetime import datetime


BEST_RESULT_PATH = "experiments/best_result.json"


def update_best_result(mean_acc, std_acc, config):

    os.makedirs("experiments", exist_ok=True)

    current_result = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "config": config,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(BEST_RESULT_PATH):
        with open(BEST_RESULT_PATH, "r") as f:
            best_result = json.load(f)

        if mean_acc > best_result["mean_accuracy"]:
            print("\n🔥 New Best Model Found! Updating record.")
            with open(BEST_RESULT_PATH, "w") as f:
                json.dump(current_result, f, indent=4)
        else:
            print("\nNo improvement over best model.")
    else:
        print("\nNo previous record found. Saving first result.")
        with open(BEST_RESULT_PATH, "w") as f:
            json.dump(current_result, f, indent=4)