import numpy as np
import torch
from model import vgg11
from torch.utils.data import DataLoader
from dataset import PowderDataset
from tqdm import tqdm
from typing import List, Tuple, Union


def testing(
    model: vgg11,
    data: np.ndarray,
    labels: np.ndarray,
    device_id: Union[int, str] = "cuda",
) -> Tuple[List[int], List[Union[float, np.ndarray]], int, int]:
    dataset = PowderDataset(inp=data, out=labels)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)

    correct, total = 0, 0
    y_preds, y_probs = [], []
    device = torch.device(f"cuda:{device_id}" if type(device_id) == int else device_id)
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.float().unsqueeze(1).to(device) / 255.0
            labels = labels.long().to(device)

            output = model(images)
            if output.shape[0] == 1:
                output = torch.sigmoid(output)
                y_probs.append(output.cpu().numpy().item())
                predicted = 1 if output > 0.5 else 0
            else:
                output = torch.softmax(output)
                y_probs.append(output.cpu().numpy())
                _, predicted = torch.max(output.data, 1)
            y_preds.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    return y_preds, y_probs, correct, total


def majority_vote(
    y_preds: List, n: int, return_counts: bool = False, return_unique: bool = False
) -> Union[List, Tuple[List, ...]]:
    y_preds_maj = np.array(y_preds).reshape(
        [n, -1]
    )  # since the segments are classified in order, they can be reshape by the number of samples to create rows for each whole sample

    num_classes_pred = []
    num_classes_counts = []
    num_classes_unique = []
    for sample in y_preds_maj:
        unique, counts = np.unique(sample, return_counts=True)
        if unique.shape[0] == 1:
            num_classes_pred.append(unique.item())
        else:
            idx = np.where(counts == np.max(counts))
            num_classes_pred.append(unique[idx].item())
        num_classes_counts.append(counts)
        num_classes_unique.append(unique)

    if return_counts and return_unique:
        return num_classes_pred, num_classes_counts, num_classes_unique
    elif return_counts:
        return num_classes_pred, num_classes_counts
    elif return_unique:
        return num_classes_pred, num_classes_unique
    else:
        return num_classes_pred


def counts_to_probs_binary(num_counts: List, pred_labels: List) -> np.ndarray:
    probs = []
    for j, sample in enumerate(num_counts):
        if sample.shape[0] == 2:
            probs.append(sample / sample.sum())
        else:
            if pred_labels[j] == 0:
                probs.append(np.array([1.0, 0.0]))
            else:
                probs.append(np.array([0.0, 1.0]))
    return np.array(probs)


def counts_to_probs_multi(
    num_counts: List, pred_labels: List, unique_labels: List
) -> np.ndarray:
    probs = []
    for j, sample in enumerate(num_counts):
        if sample.shape[0] == 3:
            probs.append(sample / sample.sum())
        elif sample.shape[0] == 2:
            if (unique_labels[j] == [0, 1]).all():
                probs.append(np.insert(sample / sample.sum(), 2, 0.0))
            elif (unique_labels[j] == [1, 2]).all():
                probs.append(np.insert(sample / sample.sum(), 1, 0.0))
            elif (unique_labels[j] == [0, 2]).all():
                probs.append(np.insert(sample / sample.sum(), 1, 0.0))
        else:
            if pred_labels[j] == 0:
                probs.append(np.array([1.0, 0.0, 0.0]))
            elif pred_labels[j] == 1:
                probs.append(np.array([0.0, 1.0, 0.0]))
            else:
                probs.append(np.array([0.0, 0.0, 1.0]))

    return np.array(probs)
