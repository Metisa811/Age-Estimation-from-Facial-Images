"""
Training and evaluation loops shared by both the regression and
classification variants of the age estimation model.
"""

import torch


def run_regression_epoch(loader, model, criterion, device, optimizer=None):
    """Run one epoch of training (if optimizer is given) or evaluation.

    Returns the mean loss (MAE, if criterion is nn.L1Loss) over the epoch.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_samples = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for imgs, ages in loader:
            imgs, ages = imgs.to(device), ages.to(device)
            preds = model(imgs)
            loss = criterion(preds, ages)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

    return total_loss / n_samples


def run_classification_epoch(loader, model, criterion, device, optimizer=None):
    """Run one epoch of training (if optimizer is given) or evaluation
    for the classification variant.

    Returns (mean_loss, accuracy) over the epoch.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_correct, n_samples = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += imgs.size(0)

    return total_loss / n_samples, n_correct / n_samples


def train_with_early_stopping(
    train_loader, val_loader, model, criterion, optimizer, device,
    epochs=20, patience=5, scheduler=None, checkpoint_path='best_model.pt',
    mode='regression',
):
    """Generic training loop with early stopping on validation loss/MAE.

    Returns the training history dict. The best checkpoint (by validation
    metric) is saved to checkpoint_path and loaded back into `model` before
    returning.
    """
    best_val_metric = float('inf')
    epochs_no_improve = 0
    history = {'train': [], 'val': []}

    epoch_fn = run_regression_epoch if mode == 'regression' else run_classification_epoch

    for epoch in range(1, epochs + 1):
        train_result = epoch_fn(train_loader, model, criterion, device, optimizer)
        val_result = epoch_fn(val_loader, model, criterion, device, optimizer=None)

        train_metric = train_result[0] if isinstance(train_result, tuple) else train_result
        val_metric = val_result[0] if isinstance(val_result, tuple) else val_result

        if scheduler is not None:
            scheduler.step(val_metric)

        history['train'].append(train_result)
        history['val'].append(val_result)
        print(f'Epoch {epoch:2d}/{epochs} | train: {train_result} | val: {val_result}')

        if val_metric < best_val_metric:
            best_val_metric = val_metric
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    model.load_state_dict(torch.load(checkpoint_path))
    return history
