import torch


def evaluate(model, loader, device, return_preds=False):

    model.eval()

    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data.x, data.edge_index, data.batch)
            prob = torch.softmax(out, dim=1)
            pred = prob.argmax(dim=1)

            correct += (pred == data.y).sum().item()

            if return_preds:
                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(data.y.cpu().tolist())
                all_probs.extend(prob[:, 1].cpu().tolist())  # probability of class 1

    acc = correct / len(loader.dataset)

    if return_preds:
        return acc, all_preds, all_labels, all_probs

    return acc