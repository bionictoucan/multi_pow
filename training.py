import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PowderDataset
from typing import Optional, Callable, Union, Dict, Tuple
import os

class Trainer:
    """
    This is the default class for defining trainers to teach neural networks
    different tasks.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    optimiser : torch.optim.Optimizer
        The optimiser to use during training.
    loss_fn : Callable
        The loss function to use during training.
    no_of_epochs : int
        The number of epochs to train for.
    batch_size : int
        The batch size to use.
    data_pth : str
        The path to the training/validation data.
    save_dir : str, optional
        The directory to save the trained models to. Default is ``"./"`` -- the
        :abbr:`CWD (current working directory)`.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        The way to adaptively change the learning rate while training. Default
        is None. For examples on how to adaptively change the learning rate
        please see
        `here <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
    device_id : int or str, optional
        Which device to perform training on. Providing an integer will point to
        the GPU with that specific ID whereas the string option can be used to
        also specify training on the CPU. When using ``nn.DataParallel`` for the
        model, setting ``device_id = "cuda"`` uses all GPUs. Default is 0 -- use
        the GPU corresponding to ID ``"cuda:0"``.
    data_parallel : bool, optional
        Whether or not to use multiple GPUs for training. Default is ``False``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        loss_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
        no_of_epochs: int,
        batch_size: int,
        data_pth: str,
        save_dir: str = "./",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device_id: Union[int, str] = 0,
        data_parallel: bool = False,
    ) -> None:

        self.device = torch.device(
            f"cuda:{device_id}" if type(device_id) == int else device_id
        )

        if data_parallel:
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model
        self.model.to(self.device)

        self.optimiser = optimiser

        self.loss_fn = loss_fn

        self.total_epochs = no_of_epochs
        self.batch_size = batch_size

        self.data_pth = data_pth

        self.save_dir = save_dir
        if not os.path.isdir(save_dir): os.mkdir(save_dir)

        self.scheduler = scheduler

        self.current_epoch = 0

    def load_data(self) -> None:
        """
        User-defined data loading instance method.
        This should populate self.train_in, self.train_out, self.val_in and self.val_out.
        """

        raise NotImplementedError("This must be user-defined!!")

    def checkpoint(self, add_info: Optional[Dict] = None) -> None:
        """
        This class method creates a checkpoint for the current epoch.

        Parameters
        ----------
        add_info : dict, optional
            An additional information to add to the checkpoint e.g. scheduler
            state dictionary. Default is ``None``.
        """

        if isinstance(self.model, torch.nn.DataParallel):
            self.chkpt = {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }
        else:
            self.chkpt = {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }

        if isinstance(add_info, dict):
            self.chkpt.update(add_info)

    def save_checkpoint(self, custom_path: Optional[str] = None) -> None:
        """
        This class method saves the current checkpoint to the save directory
        defined when instantiating the class.

        Parameters
        ----------
        custom_path : str, optional
            Path to save the model to. Default is ``None``.
        """

        current_epoch = str(self.current_epoch).zfill(3)
        if isinstance(custom_path, str):
            save_pth = f"{custom_path}{current_epoch}.pth"
        else:
            save_pth = f"{self.save_dir}{current_epoch}.pth"

        if not hasattr(self, "chkpt"):
            self.checkpoint()
        torch.save(self.chkpt, save_pth)

    def load_checkpoint(self, filename: str) -> None:
        """
        This class method loads a checkpoint for the model.

        Parameters
        ----------
        filename : str
            The path to the model to load.
        """

        if os.path.isfile(filename):
            print(f"=> loading checkpoint at {filename}")
            chkpt = torch.load(filename)
            self.current_epoch = chkpt["epoch"]
            self.train_losses = chkpt["train_losses"]
            self.val_losses = chkpt["val_losses"]
            self.model.load_state_dict(chkpt["model_state_dict"])
            self.optimiser.load_state_dict(chkpt["optimiser_state_dict"])

            def_keys = [
                "epoch",
                "train_losses",
                "val_losses",
                "model_state_dict",
                "optimiser_state_dict",
            ]
            if len(chkpt.keys()) != 5:
                self.add_info = {}
                for key in chkpt.keys():
                    if key not in def_keys:
                        self.add_info.update({key: chkpt[key]})
            if self.scheduler:
                self.scheduler.load_state_dict(self.add_info["scheduler_state_dict"])

            train_l = self.train_losses[-1]
            val_l = self.val_losses[-1]
            print(
                f"=> loaded checkpoint at {filename} at epoch {self.current_epoch} with training loss {train_l} and validation loss {val_l}."
            )
        else:
            print(f"=> no checkpoint found at {filename}")

class BinaryTrainer(Trainer):
    def load_data(self, train_in: np.ndarray, train_out: np.ndarray, val_in: np.ndarray, val_out: np.ndarray):
        self.train_loader = DataLoader(PowderDataset(train_in, train_out), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(PowderDataset(val_in, val_out), batch_size=self.batch_size, shuffle=True)
        
    def train(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        The function used to train the classifier model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The data to train the model on wrapped up in a `PyTorch DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the training data.
        """
        self.model.train()

        batch_losses = []
        total, correct = 0.0, 0.0
        for j, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.float().to(self.device), labels.to(
                self.device
            )

            self.optimiser.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimiser.step()

            batch_losses.append(loss.item())

            # work out the % correct per epoch accumulatively
            output = output.squeeze() #gets rid of the second dimension which is always 1
            output = torch.sigmoid(output) #finds the probability from the logit
            predicted = torch.tensor([1 if x >= 0.5 else 0 for x in output]).to(self.device)
            total += labels.size(0)
            correct += (
                predicted == labels.squeeze()
            ).sum()  # produces a boolean tensor which when summed evaluates True to 1 and False to 0.

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)

    def validation(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        The function used to validate the classifier model.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            The data to validate the model on wrapped up in a `PyTorch
            DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the validation data.
        """
        self.model.eval()

        total, correct = 0.0, 0.0
        with torch.no_grad():
            batch_losses = []
            for images, labels in val_loader:
                images, labels = images.float().to(self.device), labels.to(
                    self.device
                )
                output = self.model(images)

                loss = self.loss_fn(output, labels)
                batch_losses.append(loss.item())
                
                output = output.squeeze() #gets rid of the second dimension which is always 1
                output = torch.sigmoid(output) #finds the probability from the logit
                predicted = torch.tensor([1 if x >= 0.5 else 0 for x in output]).to(self.device)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum()

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)
        
    def shady_guy(self, load = None, load_pth = "."):
        if load:
            print("=> a model is being loaded.")
            self.load_checkpoint(load_pth)
            
        #initialisation of the plotting environment
        fig = plt.figure(figsize=(6,6))
        train_ax = fig.add_subplot(2,1,1)
        val_ax = train_ax.twinx()
        train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
        train_ax.tick_params(axis="x", labelbottom=False)
        val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
        train_ax_perc = fig.add_subplot(2,1,2)
        val_ax_perc = train_ax_perc.twinx()
        train_ax_perc.set_ylabel("Training Percentage [%]", color=pt_vibrant["cyan"])
        train_ax_perc.set_xlabel("Number of Epochs")
        val_ax_perc.set_ylabel("Validation Percentage [%]", color=pt_vibrant["magenta"])
        fig.show()
        fig.canvas.draw()
        
        train_losses, val_losses, train_perc, val_perc = [], [], [], []
        
        t_init = time()
        for n in range(self.total_epochs):
            if n != 0:
                self.current_epoch += 1
            if n == 0 and load:
                self.current_epoch += 1
                
            tl, tp = self.train(train_loader=self.train_loader)
            train_losses.append(tl.item())
            train_perc.append(tp)
            
            vl, vp = self.validation(val_loader=self.val_loader)
            val_losses.append(vl.item())
            val_perc.append(vp)
            t_now = round(time() - t_init, 3)

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(vl)
                else:
                    self.scheduler.step()
            
            self.train_losses = train_losses
            self.val_losses = val_losses
            self.train_perc = train_perc
            self.val_perc = val_perc
            
            if self.scheduler:
                self.checkpoint(add_info={"scheduler_state_dict" : self.scheduler.state_dict(), "train_perc" : self.train_perc, "val_perc" : self.val_perc})
            else:
                self.checkpoint(add_info={"train_perc" : self.train_perc, "val_perc" : self.val_perc})
            self.save_checkpoint()
            
        #plot the results
            fig.suptitle(f"Time elapsed {t_now}s after epoch {self.current_epoch}")
            train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
            train_ax.tick_params(axis="x", labelbottom=False)
            train_ax_perc.set_xlabel("Number of Epochs")
            val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
            train_ax_perc.set_ylabel("Training Percentage [%]", color=pt_vibrant["cyan"])
            val_ax_perc.set_ylabel("Validation Percentage [%]", color=pt_vibrant["magenta"])
            train_ax.plot(train_losses, color=pt_vibrant["cyan"], marker="o")
            val_ax.plot(val_losses, color=pt_vibrant["magenta"], marker="o")
            train_ax_perc.plot(train_perc, color=pt_vibrant["cyan"], marker="o")
            val_ax_perc.plot(val_perc, color=pt_vibrant["magenta"], marker="o")
            fig.canvas.draw()

class MultiTrainer(Trainer):
    def train(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        The function used to train the classifier model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The data to train the model on wrapped up in a `PyTorch DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the training data.
        """
        self.model.train()

        batch_losses = []
        total, correct = 0.0, 0.0
        for j, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.float().to(self.device), labels.to(
                self.device
            )

            self.optimiser.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimiser.step()

            batch_losses.append(loss.item())

            # work out the % correct per epoch accumulatively
            _, predicted = torch.max(
                output.data, 1
            )  # output will have dimensions (batch_size, output_features) with the maximum taken along the output_features axis i.e. the probability distribution of classes, predicted is populated by the indices of the maximum in this dimension -- the class label.
            total += labels.size(0)
            correct += (
                predicted == labels
            ).sum()  # produces a boolean tensor which when summed evaluates True to 1 and False to 0.

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)

    def validation(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        The function used to validate the classifier model.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            The data to validate the model on wrapped up in a `PyTorch
            DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the validation data.
        """
        self.model.eval()

        total, correct = 0.0, 0.0
        with torch.no_grad():
            batch_losses = []
            for images, labels in val_loader:
                images, labels = images.float().to(self.device), labels.to(
                    self.device
                )
                output = self.model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

                loss = self.loss_fn(output, labels)
                batch_losses.append(loss.item())

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)

    def load_data(self, train_in, train_out, val_in, val_out):
        self.train_loader = DataLoader(PowderDataset(train_in, train_out), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(PowderDataset(val_in, val_out), batch_size=self.batch_size, shuffle=True)
        
    def shady_guy(self, load = None, load_pth = "."):
        if load:
            print("=> a model is being loaded.")
            self.load_checkpoint(load_pth)
            
        if not os.path.isdir("vgg11_balanced_9"): os.mkdir("vgg11_balanced_9")

        self.save_dir = "vgg11_balanced_9/"
        
        #initialisation of the plotting environment
        fig = plt.figure(figsize=(6,6))
        train_ax = fig.add_subplot(2,1,1)
        val_ax = train_ax.twinx()
        train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
        train_ax.tick_params(axis="x", labelbottom=False)
        val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
        train_ax_perc = fig.add_subplot(2,1,2)
        val_ax_perc = train_ax_perc.twinx()
        train_ax_perc.set_ylabel("Training Percentage [%]", color=pt_vibrant["cyan"])
        train_ax_perc.set_xlabel("Number of Epochs")
        val_ax_perc.set_ylabel("Validation Percentage [%]", color=pt_vibrant["magenta"])
        fig.show()
        fig.canvas.draw()
        
        train_losses, val_losses, train_perc, val_perc = [], [], [], []
        
        t_init = time()
        for n in range(self.total_epochs):
            if n != 0:
                self.current_epoch += 1
            if n == 0 and load:
                self.current_epoch += 1
                
            tl, tp = self.train(train_loader=self.train_loader)
            train_losses.append(tl.item())
            train_perc.append(tp)
            
            vl, vp = self.validation(val_loader=self.val_loader)
            val_losses.append(vl.item())
            val_perc.append(vp)
            t_now = round(time() - t_init, 3)

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(vl)
                else:
                    self.scheduler.step()
            
            self.train_losses = train_losses
            self.val_losses = val_losses
            self.train_perc = train_perc
            self.val_perc = val_perc
            
            if self.scheduler:
                self.checkpoint(add_info={"scheduler_state_dict" : self.scheduler.state_dict(), "train_perc" : self.train_perc, "val_perc" : self.val_perc})
            else:
                self.checkpoint(add_info={"train_perc" : self.train_perc, "val_perc" : self.val_perc})
            self.save_checkpoint()
            
        #plot the results
            fig.suptitle(f"Time elapsed {t_now}s after epoch {self.current_epoch}")
            train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
            train_ax.tick_params(axis="x", labelbottom=False)
            train_ax_perc.set_xlabel("Number of Epochs")
            val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
            train_ax_perc.set_ylabel("Training Percentage [%]", color=pt_vibrant["cyan"])
            val_ax_perc.set_ylabel("Validation Percentage [%]", color=pt_vibrant["magenta"])
            train_ax.plot(train_losses, color=pt_vibrant["cyan"], marker="o")
            val_ax.plot(val_losses, color=pt_vibrant["magenta"], marker="o")
            train_ax_perc.plot(train_perc, color=pt_vibrant["cyan"], marker="o")
            val_ax_perc.plot(val_perc, color=pt_vibrant["magenta"], marker="o")
            fig.canvas.draw()