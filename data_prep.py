import numpy as np
import pandas as pd
from skimage.io import imread
import os
from tqdm import tqdm
from utils import segmentation
from typing import Optional, Tuple, List


def training_data_prep(
    imgs_pth: str,
    class_pth: str,
    model_classes: str = "cohesive",
    augs: bool = True,
    test_imgs: Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will take the image directory, CSV file containing the classes
    and what kind of model the data is to be made for and return the full
    training/validation dataset and its associated labels.

    Parameters
    ----------
    imgs_pth : str
        The path to the directory of large images.
    class_pth : str
        The path to the file containing the class labels.
    model_classes : str, optional
        How to generate the class labels for the model. Default is "cohesive".
        This will generate labels for a binary model of cohesive vs.
        non-cohesive where everything cohesive will be assigned 0 while
        non-cohesive is assigned 1. Other options include "free flowing", which
        will assign 1 to all free flowing images and 0 to the rest, and "multi"
        which will generate the multiclass labels where 0 refers to cohesive
        materials, 1 to easy flowing and 2 to free flowing.
    augs : bool, optional
        Whether or not to apply the 2 augmentations used on the data
        in the paper. These are the data flipped vertically and the data flipped
        horizontally. Default is True.
    test_imgs: list, optional
        Which images to exclude from the training and validation data for use as
        an external test set. Default is None which uses the list of images used
        for the external test set in the paper.

    Returns
    -------
    segmented_trainval : np.ndarray
        The training and validation datasets prepared for training the model.
    all_classes : np.ndarray
        The class labels for each segment in the training and validation dataset.
    """

    imgs = sorted(
        [imgs_pth + x for x in os.listdir(imgs_pth) if ".jpg" in x]
    )  # creates a list of all image paths in alphabetical order

    df = pd.read_csv(class_pth, delimiter="\t")
    sort_df = df.sort_values(
        "imagename"
    ).reset_index()  # reorders the .txt file by the `imagename` column, alphabetically

    if not isinstance(
        test_imgs, list
    ):  # if the test image paths are not specified then use the ones used in the paper
        test_imgs = [
            "Alcohol cetyl.jpg",
            "Calcium Carbonate (40%) - multicomponent.jpg",
            "Ibuprofen 50.jpg",
            "Avicel PH-101.jpg",
            "Soluplus.jpg",
            "Mefenamic acid.jpg",
            "Pearlitol 200SD.jpg",
            "Lidocaine.jpg",
            "Dimethyl fumarate.jpg",
        ]

    trainval_df = sort_df.loc[
        ~sort_df["imagename"].isin(test_imgs)
    ]  # get rid of the image paths for the external testing images
    if "Span 60" in trainval_df.Material:  # get rid of Span 60
        trainval_df.drop(
            trainval_df[trainval_df.Material == "Span 60"].index, inplace=True
        )

    classes = trainval_df["Class"].to_numpy()
    num_classes = np.zeros(classes.shape[0], dtype=np.uint8)
    for j, c in enumerate(classes):
        if model_classes == "cohesive":
            if c.lower() == "cohesive":
                num_classes[j] = 0
            elif c.lower() == "easyflowing":
                num_classes[j] = 1
            elif c.lower() == "freeflowing":
                num_classes[j] = 1
        elif model_classes == "free flowing":
            if c.lower() == "cohesive":
                num_classes[j] = 0
            elif c.lower() == "easyflowing":
                num_classes[j] = 0
            elif c.lower() == "freeflowing":
                num_classes[j] = 1
        elif model_classes == "multi":
            if c.lower() == "cohesive":
                num_classes[j] = 0
            elif c.lower() == "easyflowing":
                num_classes[j] = 1
            elif c.lower() == "freeflowing":
                num_classes[j] = 2

    trainval_rows = trainval_df.index
    trainval_imgs = [imgs[ind] for ind in trainval_rows]

    aug_imgs = []
    if augs:
        for j, img in enumerate(tqdm(trainval_imgs)):
            img = imread(img)
            img_v = img[::-1]  # flipped vertically
            img_h = img[:, ::-1]  # flipped horizontally

            aug_imgs.extend([img, img_v, img_h])
    else:
        for j, img in enumerate(tqdm(trainval_imgs)):
            aug_imgs.append(imread(img))

    for j, img in enumerate(tqdm(aug_imgs)):
        if j == 0:
            segments = segmentation(img, n=1024)
            segmented_trainval = np.expand_dims(segments, axis=0)
        else:
            segments = segmentation(img, n=1024)
            segments = np.expand_dims(segments, axis=0)
            segmented_trainval = np.append(segmented_trainval, segments, axis=0)

    no_segments = segmented_trainval.shape[1]
    segmented_trainval = segmented_trainval.reshape([-1, 1024, 1024])

    if augs:
        classeslists = [[x] * 3 * no_segments for x in num_classes]
    else:
        classeslists = [
            [x] * no_segments for x in num_classes
        ]  # if there are augmentations we must multiply the number of labels by the number of augmentations, also need to increase the number of labels by the number of segments

    all_classes = np.array([lab for lst in classeslists for lab in lst])

    assert (
        segmented_trainval.shape[0] == all_classes.shape[0]
    )  # make sure the input and output data have the same dimensions

    return segmented_trainval, all_classes


def testing_data_prep(
    imgs_pth: str,
    class_pth: str,
    model_classes: str = "cohesive",
    test_imgs: Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function will take the image directory, CSV file containing the classes
    and what kind of model the data is to be made for and return the full
    training/validation dataset and its associated labels.

    Parameters
    ----------
    imgs_pth : str
        The path to the directory of images.
    class_pth : str
        The path to the file containing the class labels.
    model_classes : str, optional
        How to generate the class labels for the model. Default is "cohesive".
        This will generate labels for a binary model of cohesive vs.
        non-cohesive where everything cohesive will be assigned 0 while
        non-cohesive is assigned 1. Other options include "free flowing", which
        will assign 1 to all free flowing images and 0 to the rest, and "multi"
        which will generate the multiclass labels where 0 refers to cohesive
        materials, 1 to easy flowing and 2 to free flowing.
    test_imgs: list, optional
        Which images to exclude from the training and validation data for use as
        an external test set. Default is None which uses the list of images used
        for the external test set in the paper.

    Returns
    -------
    segmented_test : np.ndarray
        The testing data prepared to test the model.
    all_classes : np.ndarray
        The class labels for each image segment.
    num_classes : np.ndarray
        The class labels for the entire samples for comparison when doing a
        majority vote on the class of the sample.
    """

    imgs = sorted(
        [imgs_pth + x for x in os.listdir(imgs_pth) if ".jpg" in x]
    )  # creates a list of all image paths in alphabetical order

    df = pd.read_csv(class_pth, delimiter="\t")
    sort_df = df.sort_values(
        "imagename"
    ).reset_index()  # reorders the .txt file by the `imagename` column, alphabetically

    if not isinstance(
        test_imgs, list
    ):  # if the test image paths are not specified then use the ones used in the paper
        test_imgs = [
            "Alcohol cetyl.jpg",
            "Calcium Carbonate (40%) - multicomponent.jpg",
            "Ibuprofen 50.jpg",
            "Avicel PH-101.jpg",
            "Soluplus.jpg",
            "Mefenamic acid.jpg",
            "Pearlitol 200SD.jpg",
            "Lidocaine.jpg",
            "Dimethyl fumarate.jpg",
        ]

    test_df = sort_df.loc[
        sort_df["imagename"].isin(test_imgs)
    ]  # get rid of the image paths for the training and validation images

    classes = test_df["Class"].to_numpy()
    num_classes = np.zeros(classes.shape[0], dtype=np.uint8)
    for j, c in enumerate(classes):
        if model_classes == "cohesive":
            if c.lower() == "cohesive":
                num_classes[j] = 0
            elif c.lower() == "easyflowing":
                num_classes[j] = 1
            elif c.lower() == "freeflowing":
                num_classes[j] = 1
        elif model_classes == "free flowing":
            if c.lower() == "cohesive":
                num_classes[j] = 0
            elif c.lower() == "easyflowing":
                num_classes[j] = 0
            elif c.lower() == "freeflowing":
                num_classes[j] = 1
        elif model_classes == "multi":
            if c.lower() == "cohesive":
                num_classes[j] = 0
            elif c.lower() == "easyflowing":
                num_classes[j] = 1
            elif c.lower() == "freeflowing":
                num_classes[j] = 2

    test_rows = test_df.index
    test_imgs = [imgs[ind] for ind in test_rows]

    aug_imgs = []
    for j, img in enumerate(tqdm(test_imgs)):
        aug_imgs.append(imread(img))

    for j, img in enumerate(aug_imgs):
        if j == 0:
            segments = segmentation(img, n=1024)
            segmented_test = np.expand_dims(segments, axis=0)
        else:
            segments = segmentation(img, n=1024)
            segments = np.expand_dims(segments, axis=0)
            segmented_test = np.append(segmented_test, segments, axis=0)

    no_segments = segmented_test.shape[1]
    segmented_test = segmented_test.reshape([-1, 1024, 1024])

    classeslists = [[x] * no_segments for x in num_classes]

    all_classes = np.array([lab for lst in classeslists for lab in lst])

    assert (
        segmented_test.shape[0] == all_classes.shape[0]
    )  # make sure the input and output data have the same dimensions

    return segmented_test, all_classes, num_classes
