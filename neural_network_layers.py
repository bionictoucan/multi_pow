import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Dict, Union

class FCLayer(nn.Module):
    """
    A modifiable fully-connected layer for deep networks.

    Parameters
    ----------
    in_nodes : int
        The number of input nodes to the fully-connected layer.
    out_nodes : int
        The number of output nodes of the fully-connected layer.
    bias : bool, optional
        Whether or not to use a bias node in the fully-connected layer. Default
        is False. i.e. whether to have the fully-connected layer perform the
        transformation as :math:`y = \Theta x` without bias or :math:`y = \Theta
        x + b` where :math:`b` is the bias of the node.
    normalisation : str, optional
        The normalisation to use in the layer. Default is None -- no
        normalisation used. Options "batch" and "instance" are supported to
        perform batch or instance normalisation on the data.
    activation : str, optional
        The nonlinearity to use for the layer. Default is "relu" -- uses the
        rectified linear unit (ReLU) nonlinearity. Other options supported are
        "leaky_relu", "sigmoid" and "tanh".
    initialisation : str, optional
        How to initialise the learnable parameters in a layer. Default is
        "kaiming" -- learnable parameters are initialised using Kaiming
        initialisation. Other options supported are "He" (which is equivalent to
        Kaiming initialisation but is sometime what it's called), "Xavier" or
        None.
    use_dropout : bool, optional
        Whether or not to apply dropout after the activation. Default is False.
    dropout_prob : float, optional
        Probability of a node dropping out of the network. Default is 0.5.
    lin_kwargs : dict, optional
        Additional keyword arguments to be passed to the `torch.nn.Linear`
        module. Default is {} -- an empty dictionary.
    norm_kwargs : dict, optional
        Additional keyword arguments to be passed to the normalisation being
        used. Default is {}.
    act_kwargs : dict, optional
        Additional keyword arguments to be passed to the activation being used.
        Default is {}.
    """

    def __init__(
        self,
        in_nodes: int,
        out_nodes: int,
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        lin_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {}
    ) -> None:
        super().__init__()

        self.lin = nn.Linear(in_nodes, out_nodes, **lin_kwargs)
        
        if (isinstance(normalisation, str)) and (normalisation.lower() == "batch"):
            self.norm = nn.BatchNorm2d(out_nodes, **norm_kwargs)
        elif (isinstance(normalisation, str)) and (normalisation.lower() == "instance"):
            self.norm = nn.InstanceNorm2d(out_nodes, **norm_kwargs)
        else:
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
            nn.init.kaiming_normal_(self.lin.weight, nonlinearity=activation)
            if bias:
                nn.init.kaiming_uniform_(self.lin.bias, nonlinearity=activation)
        elif initialisation.lower() == "xavier":
            nn.init.xavier_normal_(self.lin.weight, gain=nn.init.calculate_gain(activation))
            if bias:
                nn.init.xavier_uniform_(self.lin.bias, gain=nn.init.calculate_gain(activation))

        if use_dropout:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None

    def forward(self, inp: torch.tensor) -> torch.tensor:
        """
        The class method defining the behaviour of the constructed layer.
        Transformations are applied to the data in the order of linear,
        normalisation, activation.

        Parameters
        ----------
        inp : torch.tensor
            The input data to be transformed by the layer.

        Returns
        -------
         : torch.tensor
             The transformed data.
        """
        out = self.lin(inp)
        if self.norm:
            out = self.norm(out)
        out = self.act(out)
        if self.dropout:
            out = self.dropout(out)

        return out

class ConvLayer(nn.Module):
    """
    A modifiable convolutional layer for deep networks.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the convolutional layer.
    out_channels : int
        The number of output channels of the convolutional layer.
    kernel : int or Sequence[int], optional
        The size of the convolutional kernel to use in the `torch.nn.Conv2d`
        linear transformation. Default is 3.
    stride : int or Sequence[int], optional
        The stride of the convolutional kernel. Default is 1.
    pad : str, optional
        The type of padding to use during the convolutional layer. Default is
        "reflect". Other options available
        `here<https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_.
    bias : bool, optional
        Whether or not to use a bias node in the fully-connected layer. Default
        is False. i.e. whether to have the fully-connected layer perform the
        transformation as :math:`y = \Theta x` without bias or :math:`y = \Theta
        x + b` where :math:`b` is the bias of the node.
    normalisation : str, optional
        The normalisation to use in the layer. Default is None -- no
        normalisation used. Options "batch" and "instance" are supported to
        perform batch or instance normalisation on the data.
    activation : str, optional
        The nonlinearity to use for the layer. Default is "relu" -- uses the
        rectified linear unit (ReLU) nonlinearity. Other options supported are
        "leaky_relu", "sigmoid" and "tanh".
    initialisation : str, optional
        How to initialise the learnable parameters in a layer. Default is
        "kaiming" -- learnable parameters are initialised using Kaiming
        initialisation. Other options supported are "He" (which is equivalent to
        Kaiming initialisation but is sometime what it's called), "Xavier" or
        None.
    upsample : bool, optional
        Whether or not the convolutional layer will be used to spatially
        upsample the data using a linear interpolation. Default is False.
    upsample_factor : int, optional
        The factor that the spatial dimensions is upsampled. Default is 2.
    use_dropout : bool, optional
        Whether or not to apply dropout after the activation. Default is False.
    dropout_prob : float, optional
        Probability of a node dropping out of the network. Default is 0.5.
    conv_kwargs : dict, optional
        The additional keyword arguments to be passed to the `torch.nn.Conv2d`
        module. Default is {} -- an empty dictionary.
    norm_kwargs : dict, optional
        Additional keyword arguments to be passed to the normalisation being
        used. Default is {}.
    act_kwargs : dict, optional
        Additional keyword arguments to be passed to the activation being used.
        Default is {}.
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
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {}
    ) -> None:
        super(ConvLayer, self).__init__()

        self.upsample = upsample
        self.upsample_factor = upsample_factor

        if isinstance(kernel, int):
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

        if (isinstance(normalisation, str)) and (normalisation.lower() == "batch"):
            self.norm = nn.BatchNorm2d(out_channels, **norm_kwargs)
        elif (isinstance(normalisation, str)) and (normalisation.lower() == "instance"):
            self.norm = nn.InstanceNorm2d(out_channels, **norm_kwargs)
        else:
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

        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_prob)
        else:
            self.dropout = None

    def forward(self, inp: torch.tensor) -> torch.tensor:
        """
        The class method defining the behaviour of the constructed layer.
        Transformations are applied to the data in the order of upsample,
        linear, normalisation, activation, dropout if selected.

        Parameters
        ----------
        inp : torch.tensor
            The input data to be transformed by the layer.

        Returns
        -------
         : torch.tensor
             The transformed data.
        """
        if self.upsample:
            inp = F.interpolate(inp, scale_factor=self.upsample_factor)
        out = self.conv(inp)
        if self.norm:
            out = self.norm(out)
        out = self.act(out)

        if self.dropout:
            out = self.dropout(out)

        return out

class ConvTranspLayer(nn.Module):
    """
    A modifiable transpose convolutional layer.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the transpose convolutional layer.
    out_channels : int
        The number of output channels of the transpose convolutional layer.
    kernel : int or Sequence[int], optional
        The size of the convolutional kernel to use in the
        `torch.nn.Conv2dTranspose` linear transformation. Default is 3.
    stride : int or Sequence[int], optional
        The stride of the convolutional kernel. Default is 1. This is also used
        to define the `output_padding` kwarg for the `torch.nn.ConvTranspose2d`
        module. `output_padding` provides implicit padding on the output of the
        transpose convolution when stride > 1 to deterministically find the
        correct output shape. For more information, please see `here<https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_.
    pad : str, optional
        The type of padding to use during the convolutional layer. Default is
        "reflect". Other options available
        `here<https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2dTranspose>`_.
    bias : bool, optional
        Whether or not to use a bias node in the fully-connected layer. Default
        is False. i.e. whether to have the fully-connected layer perform the
        transformation as :math:`y = \Theta x` without bias or :math:`y = \Theta
        x + b` where :math:`b` is the bias of the node.
    normalisation : str, optional
        The normalisation to use in the layer. Default is None -- no
        normalisation used. Options "batch" and "instance" are supported to
        perform batch or instance normalisation on the data.
    activation : str, optional
        The nonlinearity to use for the layer. Default is "relu" -- uses the
        rectified linear unit (ReLU) nonlinearity. Other options supported are
        "leaky_relu", "sigmoid" and "tanh".
    initialisation : str, optional
        How to initialise the learnable parameters in a layer. Default is
        "kaiming" -- learnable parameters are initialised using Kaiming
        initialisation. Other options supported are "He" (which is equivalent to
        Kaiming initialisation but is sometime what it's called), "Xavier" or
        None.
    use_dropout : bool, optional
        Whether or not to apply dropout after the activation. Default is False.
    dropout_prob : float, optional
        Probability of a node dropping out of the network. Default is 0.5.
    conv_kwargs : dict, optional
        The additional keyword arguments to be passed to the `torch.nn.Conv2d`
        module. Default is {} -- an empty dictionary.
    norm_kwargs : dict, optional
        Additional keyword arguments to be passed to the normalisation being
        used. Default is {}.
    act_kwargs : dict, optional
        Additional keyword arguments to be passed to the activation being used.
        Default is {}.
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
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {}
    ) -> None:
        super(ConvTranspLayer, self).__init__()

        if isinstance(kernel, int):
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
            output_padding=stride//2,
            **conv_kwargs
        )

        if (isinstance(normalisation, str)) and (normalisation.lower() == "batch"):
            self.norm = nn.BatchNorm2d(out_channels, **norm_kwargs)
        elif (isinstance(normalisation, str)) and (normalisation.lower() == "instance"):
            self.norm = nn.InstanceNorm2d(out_channels, **norm_kwargs)
        else:
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

        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_prob)
        else:
            self.dropout = None

    def forward(self, inp: torch.tensor) -> torch.tensor:
        """
        The class method defining the behaviour of the constructed layer.
        Transformations are applied to the data in the order of
        linear, normalisation, activation.

        Parameters
        ----------
        inp : torch.tensor
            The input data to be transformed by the layer.

        Returns
        -------
         : torch.tensor
             The transformed data.
        """
        out = self.conv(inp)
        if self.norm:
            out = self.norm(out)
        out = self.act(out)

        if self.dropout:
            out = self.dropout(out)

        return out

class ResLayer(nn.Module):
    """
    A modifiable residual layer for deep neural networks.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the convolutional layer.
    out_channels : int
        The number of output channels of the convolutional layer.
    kernel : int or Sequence[int], optional
        The size of the convolutional kernel to use in the `torch.nn.Conv2d`
        linear transformation. Default is 3.
    stride : int or Sequence[int], optional
        The stride of the convolutional kernel. Default is 1.
    pad : str, optional
        The type of padding to use during the convolutional layer. Default is
        "reflect". Other options available
        `here<https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`_.
    bias : bool, optional
        Whether or not to use a bias node in the fully-connected layer. Default
        is False. i.e. whether to have the fully-connected layer perform the
        transformation as :math:`y = \Theta x` without bias or :math:`y = \Theta
        x + b` where :math:`b` is the bias of the node.
    normalisation : str, optional
        The normalisation to use in the layer. Default is None -- no
        normalisation used. Options "batch" and "instance" are supported to
        perform batch or instance normalisation on the data.
    activation : str, optional
        The nonlinearity to use for the layer. Default is "relu" -- uses the
        rectified linear unit (ReLU) nonlinearity. Other options supported are
        "leaky_relu", "sigmoid" and "tanh".
    initialisation : str, optional
        How to initialise the learnable parameters in a layer. Default is
        "kaiming" -- learnable parameters are initialised using Kaiming
        initialisation. Other options supported are "He" (which is equivalent to
        Kaiming initialisation but is sometime what it's called), "Xavier" or
        None.
    upsample : bool, optional
        Whether or not the convolutional layer will be used to spatially
        upsample the data using a linear interpolation. Default is False.
    upsample_factor : int, optional
        The factor that the spatial dimensions is upsampled. Default is 2.
    use_dropout : bool, optional
        Whether or not to apply dropout after the first activation. Default is False.
    dropout_prob : float, optional
        Probability of a node dropping out of the network. Default is 0.5.
    conv_kwargs : dict, optional
        The additional keyword arguments to be passed to the `torch.nn.Conv2d`
        module. Default is {} -- an empty dictionary.
    norm_kwargs : dict, optional
        Additional keyword arguments to be passed to the normalisation being
        used. Default is {}.
    act_kwargs : dict, optional
        Additional keyword arguments to be passed to the activation being used.
        Default is {}.
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

        if isinstance(kernel, int):
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

        if (isinstance(normalisation, str)) and (normalisation.lower() == "batch"):
            self.norm1 = nn.BatchNorm2d(out_channels, **norm_kwargs)
            self.norm2 = nn.BatchNorm2d(out_channels, **norm_kwargs)
        elif (isinstance(normalisation, str)) and (normalisation.lower() == "instance"):
            self.norm1 = nn.InstanceNorm2d(out_channels, **norm_kwargs)
            self.norm2 = nn.InstanceNorm2d(out_channels, **norm_kwargs)
        else:
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
            self.dropout = nn.Dropout2d(dropout_prob)
        else:
            self.dropout = None

    def forward(self, inp: torch.tensor) -> torch.tensor:
        """
        The class method defining the behaviour of the constructed layer.
        Transformations are applied to the data in the order of
        linear, normalisation, activation, dropout (if selected), linear
        normalisation, adding the input, activation.

        Parameters
        ----------
        inp : torch.tensor
            The input data to be transformed by the layer.

        Returns
        -------
         : torch.tensor
             The transformed data.
        """
        identity = inp.clone()

        if self.upsample:
            identity = F.interpolate(identity, scale_factor=self.upsample_factor)
            inp = F.interpolate(inp, scale_factor=self.upsample_factor)

        out = self.conv1(inp)
        if self.norm1:
            out = self.norm1(out)
        out = self.act(out)

        if self.dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        if self.norm2:
            out = self.norm2(out)
        
        if self.downsample:
            identity = self.downsample(identity)

        out = out + identity
        out = self.act(out)

        return out