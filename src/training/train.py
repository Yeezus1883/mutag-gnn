import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)