import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, List, Tuple, Dict, Union, Callable

class ConvLayer(nn.Module):
    """
    A modifiable convolutional layer for deep networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int,Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        pad: str = "reflect",
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        upsample: bool = False,
        upsample_factor: int = 2,
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {}
    ) -> None:
        super(ConvLayer, self).__init__()

        self.upsample = upsample
        self.upsample_factor = upsample_factor

        if type(kernel) == int:
            padding = (kernel-1)//2
        else:
            padding = [(x-1)//2 for x in kernel]
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            bias=bias,
            padding=padding,
            padding_mode=pad,
            **conv_kwargs
        )

        if (type(normalisation) == str) and (normalisation.lower() == "batch"):
            self.norm = nn.BatchNorm2d(out_channels, **norm_kwargs)
        elif (type(normalisation) == str) and (normalisation.lower() == "instance"):
            self.norm = nn.InstanceNorm2d(out_channels, **norm_kwargs)
        elif normalisation == None:
            self.norm = None

        if activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            self.act = nn.LeakyReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation.lower() == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError("Pester John to add this.")

        if initialisation.lower() == "kaiming" or "he":
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity=activation)
            if bias:
                nn.init.kaiming_uniform_(self.conv.bias, nonlinearity=activation)
        elif initialisation.lower() == "xavier":
            nn.init.xavier_normal_(self.conv.weight)
            if bias:
                nn.init.xavier_uniform_(self.conv.bias)

    def forward(self, inp: torch.tensor) -> torch.tensor:
        if self.upsample:
            inp = F.interpolate(inp, scale_factor=self.upsample_factor)
        out = self.conv(inp)
        if self.norm != None:
            out = self.norm(out)
        out = self.act(out)

        return out

class ConvTranspLayer(nn.Module):
    """
    A modifiable transpose convolutional layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int,Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        pad: str = "reflect",
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {}
    ) -> None:
        super(ConvTranspLayer, self).__init__()

        if type(kernel) == int:
            padding = (kernel - 1)//2
        else:
            padding = [(x-1)//2 for x in kernel]

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            bias=bias,
            padding=padding,
            padding_mode=pad,
            **conv_kwargs
        )

        if (type(normalisation) == str) and (normalisation.lower() == "batch"):
            self.norm = nn.BatchNorm2d(out_channels, **norm_kwargs)
        elif (type(normalisation) == str) and (normalisation.lower() == "instance"):
            self.norm = nn.InstanceNorm2d(out_channels, **norm_kwargs)
        elif normalisation == None:
            self.norm = None

        if activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            self.act = nn.LeakyReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation.lower() == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError("Pester John to add this.")

        if initialisation.lower() == "kaiming" or "he":
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity=activation)
            if bias:
                nn.init.kaiming_uniform_(self.conv.bias, nonlinearity=activation)
        elif initialisation.lower() == "xavier":
            nn.init.xavier_normal_(self.conv.weight)
            if bias:
                nn.init.xavier_uniform_(self.conv.bias)

    def forward(self, inp: torch.tensor) -> torch.tensor:
        out = self.conv(inp)
        if self.norm != None:
            out = self.norm(out)
        out = self.act(out)

        return out

class ResLayer(nn.Module):
    """
    A modifiable residual layer for deep neural networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        pad: str = "reflect",
        bias : bool = False,
        normalisation : Optional[str] = None,
        activation : str = "relu",
        initialisation : str = "kaiming",
        upsample: bool = False,
        upsample_factor: int = 2,
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {}
    ) -> None:
        super(ResLayer, self).__init__()

        self.upsample = upsample
        self.upsample_factor = upsample_factor

        if type(kernel) == int:
            padding = (kernel-1)//2
        else:
            padding = [(x-1)//2 for x in kernel]

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            bias=bias,
            padding=padding,
            padding_mode=pad,
            **conv_kwargs
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel,
            stride=1,
            bias=bias,
            padding=padding,
            padding_mode=pad,
            **conv_kwargs
        )

        if (type(normalisation) == str) and (normalisation.lower() == "batch"):
            self.norm1 = nn.BatchNorm2d(out_channels, **norm_kwargs)
            self.norm2 = nn.BatchNorm2d(out_channels, **norm_kwargs)
        elif (type(normalisation) == str) and (normalisation.lower() == "instance"):
            self.norm1 = nn.InstanceNorm2d(out_channels, **norm_kwargs)
            self.norm2 = nn.InstanceNorm2d(out_channels, **norm_kwargs)
        elif normalisation == None:
            self.norm1 = None
            self.norm2 = None

        if activation.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            self.act = nn.LeakyReLU(inplace=True, **act_kwargs)
        elif activation.lower() == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation.lower() == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError("Pester John to add this.")

        if initialisation.lower() == "kaiming" or "he":
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity=activation)
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity=activation)
            if bias:
                nn.init.kaiming_uniform_(self.conv1.bias, nonlinearity=activation)
                nn.init.kaiming_uniform_(self.conv2.bias, nonlinearity=activation)
        elif initialisation.lower() == "xavier":
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.xavier_normal_(self.conv2.weight)
            if bias:
                nn.init.xavier_uniform_(self.conv1.bias)
                nn.init.xavier_uniform_(self.conv2.bias)

        #if the number of channels is changing and there is not an upsample then self.downsample is needed to transform the identity of the residual layer to the dimensions of the output so they can be added
        #if the number of channels is changing and there is also upsampling then self.downsample is needed to transform the number of channels of the identity of the residual layer with the upscaling of the identity taking place elsewhere
        #if the number of channels stays the same then the 1x1 convolution is not needed
        if in_channels != out_channels and not upsample:
            self.downsample = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )
        elif in_channels != out_channels and upsample:
            self.downsample = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.downsample = None

        if use_dropout:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

    def forward(self, inp: torch.tensor) -> torch.tensor:
        identity = inp.clone()

        if self.upsample:
            identity = F.interpolate(identity, scale_factor=self.upsample_factor)
            inp = F.interpolate(inp, scale_factor=self.upsample_factor)

        out = self.conv1(inp)
        if self.norm1 != None:
            out = self.norm1(out)
        out = self.act(out)

        if self.dropout != None:
            out = self.dropout(out)

        out = self.conv2(out)
        if self.norm2 != None:
            out = self.norm2(out)
        
        if self.downsample != None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.act(out)

        return out