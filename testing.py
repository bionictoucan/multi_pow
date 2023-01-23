import numpy as np
import torch
import torch.nn.functional as F
from model import vgg11
from torch.utils.data import DataLoader
import torchvision.models as models
from dataset import PowderDataset
from tqdm import tqdm
from typing import List, Tuple, Union


def testing(
    model: models.vgg.VGG,
    data: np.ndarray,
    labels: np.ndarray,
    device_id: Union[int, str] = "cuda",
) -> Tuple[List[int], List[Union[float, np.ndarray]], int, int]:
    """
    The function used to classify the segments of images and compare them with
    their ground truth labels.

    Parameters
    ----------
    model : torchvision.models.vgg.VGG
        The trained model to evaluate the images using.
    data : numpy.ndarray
        The images to classify.
    labels : numpy.ndarray
        The ground truth labels of the images to classify.
    device_id : int or str
        The device to use for testing. Can be an integer to specify the GPU to
        use or can be a string such as "cuda" or "cpu". Default is "cuda" i.e.
        will use the first GPU it finds available.

    Returns
    -------
    y_preds : list
        The model predictions for each image segment.
    y_probs : list
        The probabilities of the class labels from the model. This will either
        be a single number for the binary models (indicating the probability of
        the image being in class 1) or an array of numbers for the multiclass
        case (indicating the probability of each class at each location).
    correct : int
        The number of correct classifications by the model.
    total : int
        The total number of classifications by the model.
    """
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
                output = F.sigmoid(output)
                y_probs.append(output.cpu().numpy().item())
                predicted = 1 if output > 0.5 else 0
            else:
                output = F.softmax(output, dim=1)
                y_probs.append(output.cpu().numpy())
                _, predicted = torch.max(output.data, 1)
            y_preds.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    return y_preds, y_probs, correct, total


def majority_vote(
    y_preds: List, n: int, return_counts: bool = False, return_unique: bool = False
) -> Union[List, Tuple[List, ...]]:
    """
    This function will work out a majority vote for a sample where there are
    predictions for each segment.

    Parameters
    ----------
    y_preds : list
        The predictions from the model for each segment.
    n : int
        The number of samples. This is used to reshape the `y_preds` list such
        that the number of samples is a dimension.
    return_counts : bool, optional
        This is the flag for returning the number of segments classified into
        each class. Default is False.
    return_unique : bool, optional
        This is the flag for returning the number of unique labels predicted by
        the network. Default is False.

    Returns
    -------
    num_classes_pred : list
        The majority vote predictions of the model.
    num_classes_counts : list, optional
        The counts for each class predicted by the model.
    num_classes_unique : list, optional
        The unique class labels predicted for each sample.
    """
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
    """
    The function for converting the counts of each segment class labels to
    probability of each class for a whole image to help calculate the area under
    the ROC curve (AUC).

    Parameters
    ----------
    num_counts : list
        The list of the the number of counts for each sample and each class.
    pred_labels : list
        The predicted labels of the samples so the code knows that when all
        segments are classified as the same class which class the sample belongs
        to.
        
    Returns
    -------
    probs : numpy.ndarray
        The probabilities for the classes of each sample based on the majority vote.
    """
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
    """
    Same function as above but for the multiclass model.

    Parameters
    ----------
    num_counts : list
        The list of the the number of counts for each sample and each class.
    pred_labels : list
        The predicted labels of the samples so the code knows that when all
        segments are classified as the same class which class the sample belongs
        to.
    unique_labels : list
        The unique labels that have been assigned in the classification so the
        function knows where to put the probabilities if not all of the classes
        are predicted for the segments.
        
    Returns
    -------
    probs : numpy.ndarray
        The probabilities for the classes of each sample based on the majority vote.
    """
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