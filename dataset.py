import torch
from torch.utils.data import Dataset
from typing import Union, Optional
import random
from torchvision.transforms import Compose

class PowderDataset(Dataset):
    """
    This is a custom torch.utils.data.Dataset for the powder images including how to handle any transformations wanted.
    """

    def __init__(self, inp: torch.tensor, out: Optional[torch.tensor] = None, transform: Union[torch.nn.Module, torch.nn.Sequential, Compose, None] = None) -> None:
        super().__init__()

        self.data = inp
        self.labels = out
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int):
        if self.labels == None:
            return self.data[index]
        else:
            if self.transform != None:
                rn = random.random()
                if rn <= 0.5:
                    if type(self.transform) == list:
                        nt = random.randint(1, len(self.transform)-1)
                        ts = []
                        for _ in range(nt):
                            ts.append(random.choice(self.transform))
                        ts = torch.nn.Sequential(*ts)

                        return ts(self.data[index]), self.labels[index]
                    else:
                        return self.transform(self.data[index]), self.labels[index]
                else:
                    return self.data[index], self.labels[index]
            else:
                return self.data[index], self.labels[index]