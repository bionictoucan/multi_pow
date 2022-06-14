import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm
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

class AE_sparsede(nn.Module):
    """
    The autoencoder model to learn a 64x64 representation of the data from a 1024x1024 patch.
    """

    def __init__(self, in_channels: int, nef: int) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, nef, stride=2, kernel_size=7, padding=3),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.Conv2d(nef, nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.Conv2d(nef, 2*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.Conv2d(2*nef, 2*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.Conv2d(2*nef, 4*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, in_channels, stride=1, kernel_size=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 4*nef, stride=1, kernel_size=1),
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4*nef, 4*nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4*nef, 2*nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.Conv2d(2*nef, 2*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2*nef, nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.Conv2d(nef, nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(nef, in_channels, stride=2, kernel_size=7, padding=3, output_padding=1),
            nn.ReLU()
        )

        for m in self.modules():
            if not AE:
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        h = self.encoder(x)
        out = self.decoder(h)

        return out

class AE_sparsesh(nn.Module):
    """
    The autoencoder model to learn a 64x64 representation of the data from a 1024x1024 patch.
    """

    def __init__(self, in_channels: int, nef: int) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, nef, stride=2, kernel_size=7, padding=3),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            # nn.Conv2d(nef, nef, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(nef),
            # nn.LeakyReLU(),
            nn.Conv2d(nef, 2*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            # nn.Conv2d(2*nef, 2*nef, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(2*nef),
            # nn.LeakyReLU(),
            nn.Conv2d(2*nef, 4*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            # nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(4*nef),
            # nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            # nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(4*nef),
            # nn.LeakyReLU(),
            nn.Conv2d(4*nef, in_channels, stride=1, kernel_size=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 4*nef, stride=1, kernel_size=1),
            # nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(4*nef),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(4*nef, 4*nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            # nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(4*nef),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(4*nef, 2*nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            # nn.Conv2d(2*nef, 2*nef, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(2*nef),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(2*nef, nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            # nn.Conv2d(nef, nef, stride=1, kernel_size=3, padding=1),
            # nn.BatchNorm2d(nef),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(nef, in_channels, stride=2, kernel_size=7, padding=3, output_padding=1),
            nn.ReLU()
        )

        for m in self.modules():
            if not AE:
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        h = self.encoder(x)
        out = self.decoder(h)

        return out

class AE_withskip(nn.Module):
    def __init__(self, in_channels: int, nef: int) -> None:
        super().__init__()
        
        self.E1 = nn.Sequential(
            nn.Conv2d(in_channels, nef, stride=2, kernel_size=7, padding=3),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.Conv2d(nef, nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU()
        )
        self.E2 = nn.Sequential(
            nn.Conv2d(nef, 2*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.Conv2d(2*nef, 2*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU()
        )
        self.E3 = nn.Sequential(
            nn.Conv2d(2*nef, 4*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU()
        )
        self.E4 = nn.Sequential(
            nn.Conv2d(4*nef, 4*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU()
        )
        self.z1 = nn.Conv2d(4*nef, in_channels, stride=1, kernel_size=1)

        self.z2 = nn.Conv2d(in_channels, 4*nef, stride=1, kernel_size=1)
        self.D1 = nn.Sequential(
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4*nef, 4*nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU()
        )
        self.D2 = nn.Sequential(
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4*nef, 2*nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU()
        )
        self.D3 = nn.Sequential(
            nn.Conv2d(2*nef, 2*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2*nef, nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU()
        )
        self.D4 = nn.Sequential(
            nn.Conv2d(nef, nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(nef, in_channels, stride=2, kernel_size=7, padding=3, output_padding=1),
            nn.LeakyReLU()
        )

        for m in self.modules():
            if not AE:
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        E1 = self.E1(x)
        E2 = self.E2(E1)
        E3 = self.E3(E2)
        E4 = self.E4(E3)

        h = self.z1(E4)

        D1 = self.z2(h) + E4
        D2 = self.D1(D1) + E3
        D3 = self.D2(D2) + E2
        D4 = self.D3(D3) + E1

        out = self.D4(D4)

        return out

    @property
    def encoder(self) -> nn.Sequential:
        return nn.Sequential(
            self.E1,
            self.E2,
            self.E3,
            self.E4,
            self.z1
        )

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
            nn.Conv2d(nef, nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.Conv2d(nef, 2*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.Conv2d(2*nef, 2*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.Conv2d(2*nef, 4*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, in_channels, stride=1, kernel_size=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 4*nef, stride=1, kernel_size=1),
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4*nef, 4*nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.Conv2d(4*nef, 4*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4*nef, 2*nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.Conv2d(2*nef, 2*nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2*nef, nef, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.Conv2d(nef, nef, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(nef, in_channels, stride=2, kernel_size=7, padding=3, output_padding=1),
            nn.LeakyReLU()
        )

        for m in self.modules():
            if not AE:
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # inp = x.clone()
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
    def __init__(
        self,
        model : torch.nn.Module,
        optimiser : torch.optim.Optimizer,
        loss_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
        no_of_epochs: int,
        batch_size: int,
        data_pth: str,
        save_dir: str = "./",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device_id: Union[int, str] = 0,
        rho: float = 0.05
        ):
        super().__init__(model, optimiser, loss_fn, no_of_epochs, batch_size, data_pth, save_dir, scheduler, device_id)
        # self.rho = rho

    def load_data(self) -> None:
        """
        This class method loads the training and validation data. This depends on the format of the data.
        """

        f = np.load(self.data_pth)

        self.train_in = f["train_in"].reshape(-1, *f["train_in"].shape[-2:])/255
        self.train_out = f["train_in"].reshape(-1, *f["train_in"].shape[-2:])/255
        self.val_in = f["val_in"].reshape(-1, *f["val_in"].shape[-2:])/255
        self.val_out = f["val_in"].reshape(-1, *f["val_in"].shape[-2:])/255

    # def kl_divergence(self, rho, rho_hat):
    #     rho_hat = torch.mean(F.sigmoid(rho_hat).view(rho_hat.size(0),-1), 1) # sigmoid because we need the probability distributions
    #     rho = torch.tensor([rho] * len(rho_hat)).to(self.device)
    #     return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

    # # define the sparse loss function
    # def sparse_loss(self, rho, images):
    #     model_children = list(self.model.children())
    #     values = images
    #     loss = 0
    #     for i in range(len(model_children)):
    #         values = model_children[i](values)
    #         loss += self.kl_divergence(rho, values)
    #     return loss

    def _layer_extract(self) -> List:
        """
        This function will extract the layers so the L1 reg will actually be
        calculated per set of learnable parameters rather than the whole network
        because of how the model is set up, shennanigans.
        """

        #this if statement exists so this function will work when using multiple
        #or single GPU
        if isinstance(self.model, nn.DataParallel):
            model = list(self.model.children()).pop()
        else:
            model = self.model

        list_ae_ch = []
        enc, dec = list(model.children())
        conv_e, norm_e, act_e = enc[::3], enc[1::3], enc[2::3]
        for j in range(len(enc) // 3):
            list_ae_ch.append(nn.Sequential(conv_e[j], norm_e[j], act_e[j]))
        list_ae_ch.append(conv_e[-1])
        list_ae_ch.append(dec[0])
        conv_d, norm_d, act_d = dec[1:-2:3], dec[2:-2:3], dec[3:-2:3]
        for j in range((len(dec) // 3) - 1):
            list_ae_ch.append(nn.Sequential(conv_d[j], norm_d[j], act_d[j]))
        list_ae_ch.append(nn.Sequential(dec[-2:]))

        return list_ae_ch

    def sparse_loss(self, images):
        loss = 0
        model_children = self._layer_extract()
        values = images
        for i in range(len(model_children)):
            values = model_children[i](values)
            loss += torch.mean(torch.abs(values))
        return loss

    def train(self, train_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        batch_losses = []
        for j, (inputs, outputs) in enumerate(tqdm(train_loader)):
            inputs, outputs = inputs.float().to(self.device), outputs.float().to(self.device)

            self.optimiser.zero_grad()
            model_outputs = self.model(inputs)
            loss = self.loss_fn(model_outputs, outputs) + 0.001*self.sparse_loss(inputs)
            loss.backward()
            self.optimiser.step()

            batch_losses.append(loss.item())
            if j == 0:
                plt_in = inputs[0,0].clone().detach().cpu().squeeze().numpy()
                plt_gen = model_outputs[0,0].clone().detach().cpu().squeeze().numpy()

        if self.scheduler:
            self.scheduler.step()

        return torch.mean(torch.tensor(batch_losses)), plt_in, plt_gen

    def validation(self, val_loader: torch.utils.data.DataLoader) -> float:
        batch_losses = []
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.float().to(self.device), outputs.float().to(self.device)
                model_outputs = self.model(inputs)
                loss = self.loss_fn(model_outputs, outputs)

                batch_losses.append(loss.item())

        return torch.mean(torch.tensor(batch_losses))

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
        fig = plt.figure(figsize=(9,9), constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=2)
        train_ax = fig.add_subplot(gs[1,:])
        val_ax = train_ax.twinx()
        # train_ax.set_yscale("log")
        train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
        train_ax.set_xlabel("Number of Epochs")
        val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])

        train_im = fig.add_subplot(gs[0,0])
        train_im.set_xticks([])
        train_im.set_yticks([])
        train_im.set_title("Input Image")
        gen_im = fig.add_subplot(gs[0,1])
        gen_im.set_xticks([])
        gen_im.set_yticks([])
        gen_im.set_title("Generated Image")
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

            tl, plt_in, plt_gen = self.train(train_loader=train_loader)
            train_losses.append(tl.item())

            vl = self.validation(val_loader=val_loader)
            val_losses.append(vl)
            t_now = round(time() - t_init, 3)

        #save the model
            self.train_losses = train_losses
            self.val_losses = val_losses

            #only save if validation loss is minimal
#             if min(val_losses) == val_losses[-1]:
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
            train_im.set_xticks([])
            train_im.set_yticks([])
            train_im.set_title("Input Image")
            train_im.imshow(plt_in, cmap="Greys_r", origin="lower")
            gen_im.set_xticks([])
            gen_im.set_yticks([])
            gen_im.set_title("Generated Image")
            gen_im.imshow(plt_gen, cmap="Greys_r", origin="lower")
            fig.canvas.draw()