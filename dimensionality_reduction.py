import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import List, Union, Optional
from torch.utils.data import DataLoader
from battle_factory import RegressorTrainer
from dataset import PowderDataset
from torchvision.transforms import Compose
from time import time

pt_vibrant = {
    "blue" : "#0077BB",
    "cyan" : "#33BBEE",
    "teal" : "#009988",
    "orange" : "#EE7733",
    "red" : "#CC3311",
    "magenta" : "#EE3377",
    "grey" : "#BBBBBB"
}

class AE(nn.Module):
    """
    The autoencoder model to learn a 64x64 representation of the data from a 1024x1024 patch.
    """

    def __init__(self, in_channels: int, nef: int) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, nef, stride=2, kernel_size=7, padding=3),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.Conv2d(nef, 2*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.Conv2d(2*nef, 4*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, in_channels, stride=1, kernel_size=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 4*nef, stride=1, kernel_size=1),
            nn.ConvTranspose2d(4*nef, 4*nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4*nef, 2*nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2*nef, nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(nef, in_channels, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.LeakyReLU()
        )

        for m in self.modules():
            if not AE:
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        h = self.encoder(x)
        out = self.decoder(h)

        return out

class AE256(nn.Module):
    """
    The autoencoder model to learn a 256x256 representation of the data from a 1024x1024 patch.
    """

    def __init__(self, in_channels: int, nef: int) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, nef, stride=2, kernel_size=7, padding=3),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.Conv2d(nef, 2*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.Conv2d(2*nef, in_channels, stride=1, kernel_size=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 2*nef, stride=1, kernel_size=1),
            nn.ConvTranspose2d(2*nef, nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(nef, in_channels, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.LeakyReLU()
        )

        for m in self.modules():
            if not AE:
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        h = self.encoder(x)
        out = self.decoder(h)

        return out

class AETrainer(RegressorTrainer):
    def load_data(self) -> None:
        """
        This class method loads the training and validation data. This depends on the format of the data.
        """

        f = np.load(self.data_pth)

        self.train_in = f["train_in"].reshape(-1, *f["train_in"].shape[-2:])/255
        self.train_out = f["train_in"].reshape(-1, *f["train_in"].shape[-2:])/255
        self.val_in = f["val_in"].reshape(-1, *f["val_in"].shape[-2:])/255
        self.val_out = f["val_in"].reshape(-1, *f["val_in"].shape[-2:])/255

    def myth_trainer(self, load: bool = False, load_pth: Optional[str] = None, transform: Union[nn.Module, nn.Sequential, Compose, List, None] = None) -> None:
        """
        This class method trains the network with the interactive plotting environment.
        """

        if load:
            print("=> a model is being loaded.")
            self.load_checkpoint(load_pth)

        # dataset and data loader creation
        train_dataset = PowderDataset(torch.from_numpy(self.train_in).unsqueeze(1), torch.from_numpy(self.train_out).unsqueeze(1), transform=transform)
        val_dataset = PowderDataset(torch.from_numpy(self.val_in).unsqueeze(1), torch.from_numpy(self.val_out).unsqueeze(1), transform=transform)
        # val_dataset = PowderDataset(torch.from_numpy(self.val_in), torch.from_numpy(self.val_out))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        #initialisation of the plotting environment
        fig = plt.figure(figsize=(6,6))
        train_ax = fig.add_subplot(1,1,1)
        val_ax = train_ax.twinx()
        # train_ax.set_yscale("log")
        train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
        train_ax.set_xlabel("Number of Epochs")
        val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
        fig.show()
        fig.canvas.draw()

        #define lists to store the different losses in
        train_losses, val_losses = [], []

        #do the training and validation
        t_init = time()
        for n in range(self.total_epochs):
            if n != 0:
                self.current_epoch += 1
            if n == 0 and load:
                self.current_epoch += 1

            tl = self.train(train_loader=train_loader)
            train_losses.append(tl.item())

            vl = self.validation(val_loader=val_loader)
            val_losses.append(vl)
            t_now = round(time() - t_init, 3)

        #save the model
            self.train_losses = train_losses
            self.val_losses = val_losses

            #only save if validation loss is minimal
            if min(val_losses) == val_losses[-1]:
                if self.scheduler:
                    self.checkpoint(add_info={"scheduler_state_dict" : self.scheduler.state_dict()})
                else:
                    self.checkpoint()
                self.save_checkpoint()

        #plot the results
            fig.suptitle(f"Time elapsed {t_now}s after epoch {self.current_epoch}")
            train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
            train_ax.set_xlabel("Number of Epochs")
            val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
            train_ax.semilogy(train_losses, color=pt_vibrant["cyan"], marker="o")
            val_ax.semilogy(val_losses, color=pt_vibrant["magenta"], marker="o")
            fig.canvas.draw()