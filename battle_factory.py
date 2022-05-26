import torch
from typing import Optional, Callable, Union, Dict
from tqdm import tqdm

class Trainer:
    """
    This is the default class for defining trainers to teach neural networks different tasks.
    """

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
        device_id: Union[int, str] = 0
    ) -> None:

        # self.device = torch.device(f"cuda:{device_id}" if type(device_id) ==
        # int else device_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.nn.DataParallel(model)
        self.model.to(self.device)

        self.optimiser = optimiser

        self.loss_fn = loss_fn

        self.total_epochs = no_of_epochs
        self.batch_size = batch_size

        self.data_pth = data_pth

        self.save_dir = save_dir

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
        """

        self.chkpt = {
            "epoch" : self.current_epoch,
            "model_state_dict" : self.model.state_dict(),
            "optimiser_state_dict" : self.optimiser.state_dict(),
            "train_losses" : self.train_losses,
            "val_losses" : self.val_losses
        }

        if add_info != None:
            self.chkpt.update(add_info)

    def save_checkpoint(self, custom_path: Optional[str] = None) -> None:
        """
        This class method saves the current checkpoint to the save directory defined when instantiating the class.
        """

        if custom_path != None:
            save_pth = f"{custom_path}{self.current_epoch}.pth"
        else:
            save_pth = f"{self.save_dir}{self.current_epoch}.pth"

        if not hasattr(self, "chkpt"):
            self.checkpoint()
        torch.save(self.chkpt, save_pth)

    def load_checkpoint(self, filename: str) -> None:
        """
        This class method loads a checkpoint for the model.
        """

        if os.path.isfile(filename):
            print(f"=> loading checkpoint at {filename}")
            chkpt = torch.load(filename)
            self.current_epoch = chkpt["epoch"]
            self.train_losses = chkpt["train_losses"]
            self.val_losses = chkpt["val_losses"]
            self.model.load_state_dict(chkpt["model_state_dict"])
            self.optimiser.load_state_dict(chkpt["optimiser_state_dict"])

            def_keys = ["epoch", "train_losses", "val_losses", "model_state_dict", "optimiser_state_dict"]
            if len(chkpt.keys()) != 5:
                self.add_info = {}
                for key in chkpt.keys():
                    if key not in def_keys:
                        self.add_info.update({key : chkpt[key]})
            if self.scheduler:
                self.scheduler.load_state_dict(self.add_info["scheduler_state_dict"])

            train_l = self.train_losses[-1]
            val_l = self.val_losses[-1]
            print(f"=> loaded checkpoint at {filename} at epoch {self.current_epoch} with training loss {train_l} and validation loss {val_l}.")
        else:
            print(f"=> no checkpoint found at {filename}")

class ClassifierTrainer(Trainer):
    """
    A trainer for training a neural network for classification.
    """
    def train(self, train_loader: torch.utils.data.DataLoader) -> float:
        self.model.train()

        batch_losses = []
        for j, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.float().to(self.device), labels.long().to(self.device)

            self.optimiser.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimiser.step()

            batch_losses.append(loss.item())

        if self.scheduler:
            self.scheduler.step()

        return torch.mean(torch.tensor(batch_losses))

    def validation(self, val_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()

        total, correct = 0., 0.
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.float().to(self.device), labels.long().to(self.device)
                output = self.model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

        return float((correct / total) * 100)

class RegressorTrainer(Trainer):
    """
    A trainer for training a neural network for regression.
    """
    def train(self, train_loader: torch.utils.data.DataLoader) -> float:
        batch_losses = []
        for j, (inputs, outputs) in enumerate(tqdm(train_loader)):
            inputs, outputs = inputs.float().to(self.device), outputs.float().to(self.device)

            self.optimiser.zero_grad()
            model_outputs = self.model(inputs)
            loss = self.loss_fn(model_outputs, outputs)
            loss.backward()
            self.optimiser.step()

            batch_losses.append(loss.item())

        if self.scheduler:
            self.scheduler.step()

        return torch.mean(torch.tensor(batch_losses))

    def validation(self, val_loader: torch.utils.data.DataLoader) -> float:
        batch_losses = []
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.float().to(self.device), outputs.float().to(self.device)
                model_outputs = self.model(inputs)
                loss = self.loss_fn(model_outputs, outputs)

                batch_losses.append(loss.item())

        return torch.mean(torch.tensor(batch_losses))