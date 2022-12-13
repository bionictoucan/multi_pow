import numpy as np
import torch
from model import vgg11
from torch.utils.data import DataLoader
from dataset import PowderDataset
from tqdm import tqdm
from typing import List, Tuple, Union

def testing(model: vgg11, data: np.ndarray, labels: np.ndarray, device_id: Union[int, str] = "cuda") -> Tuple[List[int], List[Union[float, np.ndarray]], int, int]:
    dataset = PowderDataset(inp=data, out=labels)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)

    correct, total = 0, 0
    y_preds, y_probs = [], []
    device = torch.device(f"cuda:{device_id}" if type(device_id) == int else device_id)
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.float().unsqueeze(1).to(device)
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