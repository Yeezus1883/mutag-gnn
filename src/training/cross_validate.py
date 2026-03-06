# import os

# import torch
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from torch_geometric.loader import DataLoader

# from src.models.gcn import GCN
# from src.training.train import train_one_epoch
# from src.training.evaluate import evaluate
# from src.models.base import get_model

# from sklearn.metrics import (
#     confusion_matrix,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     classification_report
# )


# def cross_validate(data_list, config, in_channels, num_classes, device):

#     best_fold_acc = 0
#     best_fold_state = None

#     labels = [data.y.item() for data in data_list]

#     skf = StratifiedKFold(
#         n_splits=10,
#         shuffle=True,
#         random_state=config["seed"]
#     )

#     fold_accuracies = []
#     all_fold_preds = []
#     all_fold_labels = []
#     all_fold_probs = []

#     for fold, (train_idx, test_idx) in enumerate(skf.split(data_list, labels)):

#         print(f"\n--- Fold {fold + 1} ---")

#         train_data = [data_list[i] for i in train_idx]
#         test_data = [data_list[i] for i in test_idx]

#         train_loader = DataLoader(
#             train_data,
#             batch_size=config["batch_size"],
#             shuffle=True
#         )

#         test_loader = DataLoader(
#             test_data,
#             batch_size=config["batch_size"]
#         )

#         model = get_model(
#         config,
#         in_channels,
#         num_classes
#         ).to(device) 


#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=config["lr"],
#             weight_decay=config["weight_decay"]
#         )
#         scheduler = None
#         if config.get("scheduler", None) == "cosine":
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             T_max=config["epochs"]
#         )

#         criterion = torch.nn.CrossEntropyLoss()

#         for epoch in range(1, config["epochs"] + 1):
#             train_one_epoch(model, train_loader, optimizer, criterion, device)
#             if scheduler is not None:
#                 scheduler.step()

#         test_acc = evaluate(model, test_loader, device)
#         if test_acc > best_fold_acc:
#             best_fold_acc = test_acc
#             best_fold_state = model.state_dict()
#         print(f"Fold {fold + 1} Accuracy: {test_acc:.4f}")

#         fold_accuracies.append(test_acc)

#     mean_acc = np.mean(fold_accuracies)
#     std_acc = np.std(fold_accuracies)

#     print("\n===== Cross-Validation Results =====")
#     print(f"Mean Accuracy: {mean_acc:.4f}")
#     print(f"Std Dev: {std_acc:.4f}")

#     # Save best fold model
#     if best_fold_state is not None:
#         os.makedirs("experiments", exist_ok=True)
#         torch.save(best_fold_state, "experiments/best_model.pt")
#         print("Best fold model saved to experiments/best_model.pt")

#     return mean_acc, std_acc

import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.models.base import get_model
from src.training.train import train_one_epoch
from src.training.evaluate import evaluate


def cross_validate(data_list, config, in_channels, num_classes, device):

    labels = [data.y.item() for data in data_list]

    skf = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=config["seed"]
    )

    fold_accuracies = []

    all_fold_preds = []
    all_fold_labels = []
    all_fold_probs = []

    best_fold_acc = 0
    best_fold_state = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_list, labels)):

        print(f"\n--- Fold {fold + 1} ---")

        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]

        train_loader = DataLoader(
            train_data,
            batch_size=config["batch_size"],
            shuffle=True
        )

        test_loader = DataLoader(
            test_data,
            batch_size=config["batch_size"]
        )

        model = get_model(
            config,
            in_channels,
            num_classes
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"])
        )

        # Optional Scheduler
        scheduler = None
        if config.get("scheduler", "none") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(config["epochs"])
            )

        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, int(config["epochs"]) + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

            if scheduler is not None:
                scheduler.step()

        test_acc, preds, labels_fold, probs = evaluate(
            model,
            test_loader,
            device,
            return_preds=True
        )

        print(f"Fold {fold + 1} Accuracy: {test_acc:.4f}")

        fold_accuracies.append(test_acc)

        all_fold_preds.extend(preds)
        all_fold_labels.extend(labels_fold)
        all_fold_probs.extend(probs)

        # Track best fold model
        if test_acc > best_fold_acc:
            best_fold_acc = test_acc
            best_fold_state = model.state_dict()

    # Save best fold model
    if best_fold_state is not None:
        os.makedirs("experiments", exist_ok=True)
        torch.save(best_fold_state, "experiments/best_model.pt")
        print("\nBest fold model saved to experiments/best_model.pt")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print("\n===== Cross-Validation Results =====")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Std Dev: {std_acc:.4f}")

    # -------- Full Classification Metrics --------
    cm = confusion_matrix(all_fold_labels, all_fold_preds)

    precision = precision_score(all_fold_labels, all_fold_preds)
    recall = recall_score(all_fold_labels, all_fold_preds)
    f1 = f1_score(all_fold_labels, all_fold_preds)
    roc_auc = roc_auc_score(all_fold_labels, all_fold_probs)

    print("\n===== Classification Metrics (Aggregated) =====")
    print("Confusion Matrix:")
    print(cm)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_fold_labels, all_fold_preds))

    return mean_acc, std_acc