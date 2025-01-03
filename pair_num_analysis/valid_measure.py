import torch
from tqdm import tqdm


def train(model, device, loader, optimizer, criterion_fn):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = criterion_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()

    return loss


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    mae = evaluator.eval(input_dict)['mae']

    return mae, y_true, y_pred


def eval_loss(model, device, loader, criterion_fn):
    model.eval()
    loss_accum = 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            loss = criterion_fn(pred, batch.y)
            loss_accum += loss.detach().cpu().item()

    return loss_accum / len(loader)

def eval_accu(model, device, loader, evaluator):
    model.eval()
    total_samples = 0
    correct_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_true = batch.y.view(pred.shape).detach().cpu()
            y_pred = torch.round(pred).detach().cpu().long()

            correct_samples += torch.sum(torch.round(y_pred) == y_true).item()
            total_samples += y_true.numel()

    # Calculate accuracy
    accuracy = correct_samples / total_samples

    return accuracy

def own_accu(model, device, loader, evaluator):
    model.eval()
    total_samples = 0
    correct_samples = 0

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model(batch).view(-1, )
            y_true = batch.y.view(pred.shape).detach().cpu()

            y_pred = pred.detach().cpu()

            error = torch.abs(y_pred - y_true)

            correct_predictions = error <= 1
            correct_samples += correct_predictions.sum().item()
            total_samples += y_true.numel()
        accuracy = correct_samples / total_samples

        return accuracy

def eval_recall(model, device, loader, num_classes):
    model.eval()
    total_samples = 0
    true_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_true = batch.y.view(pred.shape).detach().cpu().long()
            y_pred = torch.round(pred).detach().cpu().long()

            # Count true positives and false negatives for each class
            for class_idx in range(num_classes):
                true_positives[class_idx] += torch.sum((y_pred == class_idx) & (y_true == class_idx)).item()
                false_negatives[class_idx] += torch.sum((y_pred != class_idx) & (y_true == class_idx)).item()

            # Update total samples
            total_samples += y_true.numel()

    recall_per_class = true_positives / (true_positives + false_negatives)

    micro_avg_recall = torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_negatives))

    return float(micro_avg_recall)

def eval_precision(model, device, loader, num_classes):
    model.eval()
    total_samples = 0
    true_positives = torch.zeros(num_classes)
    false_positive = torch.zeros(num_classes)

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_true = batch.y.view(pred.shape).detach().cpu().long()
            y_pred = torch.round(pred).detach().cpu().long()

            # Count true positives and false negatives for each class
            for class_idx in range(num_classes):
                true_positives[class_idx] += torch.sum((y_pred == class_idx) & (y_true == class_idx)).item()
                false_positive[class_idx] += torch.sum((y_pred == class_idx) & (y_true != class_idx)).item()

            # Update total samples
            total_samples += y_true.numel()

    perprecision_per_class  = true_positives / (true_positives + false_positive)

    micro_avg_precision = torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_positive))

    return float(micro_avg_precision)


def eval_f1_score(model, device, loader, num_classes):
    model.eval()
    total_samples = 0
    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model(batch).view(-1, )
            y_true = batch.y.view(pred.shape).detach().cpu().long()
            y_pred = torch.round(pred).detach().cpu().long()

            # Count true positives, false positives, and false negatives for each class
            for class_idx in range(num_classes):
                true_positives[class_idx] += torch.sum((y_pred == class_idx) & (y_true == class_idx)).item()
                false_positives[class_idx] += torch.sum((y_pred == class_idx) & (y_true != class_idx)).item()
                false_negatives[class_idx] += torch.sum((y_pred != class_idx) & (y_true == class_idx)).item()

            # Update total samples
            total_samples += y_true.numel()

    # Calculate precision, recall, and F1 score for each class
    precision_per_class = true_positives / (true_positives + false_positives)
    recall_per_class = true_positives / (true_positives + false_negatives)

    # Avoid division by zero
    precision_per_class[torch.isnan(precision_per_class)] = 0.0
    recall_per_class[torch.isnan(recall_per_class)] = 0.0

    # Calculate F1 score for each class
    f1_score_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)

    # Calculate micro-average precision, recall, and F1 score
    micro_avg_precision = torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_positives))
    micro_avg_recall = torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_negatives))
    micro_avg_f1_score = 2 * (micro_avg_precision * micro_avg_recall) / (micro_avg_precision + micro_avg_recall)

    # return f1_score_per_class, micro_avg_f1_score
    return float(micro_avg_f1_score)

def test(model, device, loader):
    model.eval()
    y_pred = []

    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_pred = y_pred.float()

    return y_pred