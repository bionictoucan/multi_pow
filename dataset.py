import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Union, Optional, Tuple, List
import random
from torchvision.transforms import Compose


class PowderDataset(Dataset):
    """
    This is a custom torch.utils.data.Dataset for the powder images including
    how to handle any transformations wanted.

    .. note::
        To provide deterministic augmentations please use Python's internal `random.seed() <https://docs.python.org/3/library/random.html#random.seed>`_ to set the random seed.

    .. warning::
        Setting the random seed *may* affect random weight initialisation and dataset splitting (if other random seeds are not explicitly stated for these steps). This is something I need to check what happens if a random seed is set by the random library different from ``numpy`` and ``PyTorch``.

    Parameters
    ----------
    inp : torch.Tensor
        The input powder images to whatever network is to be trained.
    out : torch.Tensor, optional
        The output of the network, can be class labels or FFc values or the
        images themselves for the autoencoder. Can also be left empty when used
        for testing. Default is ``None``.
    transform : torch.nn.Module, torch.nn.Sequential, torchvision.transforms.Compose, list, Optional
        Any data augmentations to be done to the input data. These
        transformations will be carried out as batches are called from the
        dataset. Transformations can take the form of ``torch.nn.Module`` for
        single augmentations and ``torch.nn.Sequential``,
        ``torchvision.transforms.Compose`` or ``list`` for multiple augmentations.

        When augmentations are defined a random number is drawn from the unit
        uniform distribution (using Python's internal `random.random() <https://docs.python.org/3/library/random.html#random.random>`_ function). If this value is <= ``aug_prob``, then the data is
        augmented. Essentially there is a (``aug_prob`` *100)% chance of data augmentation.

        When the augmentations are given as a ``torch.nn.Module``,
        ``torch.nn.Sequential`` or ``torchvision.transforms.Compose`` then the
        augmentations are applied in their current form to the data. If the
        augmentations are given as a list, the ``aug_type`` kwarg defines the
        behaviour of the augmentations. For ``aug_type = "single"``, one
        transformation is selected randomly from the list (by sampling the
        uniform distribution between ``[1, len(transform)]``) and applied to the
        data. For ``aug_type = "multi"``, the list of augmentations is sampled a
        random number of times (by sampling the same uniform distribution as
        before), e.g. if the random number chosen is 3, then 3 augmentations
        would be chosen at random and applied to the data.

        .. note::
           This method will NOT pick the same augmentation twice.

    aug_prob : float, optional
        The probability threshold set such that if a random number drawn from
        the uniform unit distribution is <= ``aug_prob`` the augmentations will
        be applied to the data. Default is 0.5.
    aug_type : str, optional
        The behaviour of the list of augmentations assuming that
        ``isinstance(self.transform, list) == True``. The user can choose from
        the two options ``["single", "multi"]``. For ``"single"``, one
        augmentation is
        chosen at random (by sampling an integer from the unit distribution
        ``[1, len(transform)]`` using Python's internal `random.randint() <https://docs.python.org/3/library/random.html#random.randint>`_
        function).

        .. note::
            This random integer is sampled *every time* the dataset is accessed meaning that each time a batch is created a different augmentation will be applied.

            In fact, this holds true for applying the augmentation in general.

        For ``"multi"``, the random integer is used to define the number of
        augmentations applied to the data. These augmentations will be chosen at
        random using Python's internal `random.sample() <https://docs.python.org/3/library/random.html#random.sample>`_ function.
    """

    def __init__(
        self,
        inp: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        transform: Union[
            torch.nn.Module, torch.nn.Sequential, Compose, List, None
        ] = None,
        aug_prob: float = 0.5,
        aug_type: str = "single",
    ) -> None:
        super().__init__()

        self.data = inp
        self.labels = out
        self.transform = transform
        self.aug_prob = aug_prob
        self.aug_type = aug_type

    def __len__(self) -> int:
        """
        Returns the number of elements in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns some data of the dataset.

        Parameters
        ----------
        index : int or Slice
            The index or range of indices at which to return data from the
            dataset.

        Returns
        -------
         : torch.Tensor or tuple
            Either the specified input data is returned if there is no
            corresponding output or a tuple of the (input, output) is returned
            with the input having been augmented if the object is set up to do so.
        """
        if not isinstance(self.labels, (torch.Tensor, np.ndarray, list)):
            return self.data[index]
        else:
            if isinstance(
                self.transform, (torch.nn.Module, torch.nn.Sequential, Compose, list)
            ):
                rn = random.random()
                if rn <= self.aug_prob:
                    if isinstance(self.transform, list):
                        nt = random.randint(1, len(self.transform))
                        if self.aug_type == "multi":
                            ts = torch.nn.Sequential(
                                *random.sample(self.transform, k=nt)
                            )
                        elif self.aug_type == "single":
                            ts = self.transform[nt - 1]

                        return ts(self.data[index]), self.labels[index]
                    else:
                        return self.transform(self.data[index]), self.labels[index]
                else:
                    return self.data[index], self.labels[index]
            else:
                return self.data[index], self.labels[index]
