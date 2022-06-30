import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Sequence
from neural_network_layers import ConvLayer, ResLayer


class CNNClassifier(nn.Module):
    """
    The class for creating a custom convolutional neural network (CNN) for
    classification.

    Parameters
    ----------
    layers : Sequence[int]
        The layout of the layers in the CNN. e.g. how many layers to a block. A
        convolutional block is constructed from convolutional layers in
        conjunction with the `in_channels`, `out_channels` and
        `intermediate_channels` kwargs. The number of blocks is to be decided by
        the user depending on how many times they would like to downsample the
        input data.
    """

    def __init__(
        self,
        layers: Sequence[int],
        num_classes: int,
        in_channels: int,
        out_channels: int,
        intermediate_channels: Sequence[int],
        kernel_sizes: Union[int, Sequence[int], Sequence[Sequence[int]]],
        strides: Union[int, Sequence[int], Sequence[Sequence[int]]],
        pad: str = "reflect",
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {},
        avgpool_size: Union[int, Sequence[int]] = 7,
        fc_nodes: int = 4096,
    ) -> None:
        super().__init__()

        self.features = self.make_feature_extractor(
            layers,
            in_channels,
            out_channels,
            intermediate_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pad=pad,
            bias=bias,
            normalisation=normalisation,
            activation=activation,
            initialisation=initialisation,
            conv_kwargs=conv_kwargs,
            norm_kwargs=norm_kwargs,
            act_kwargs=act_kwargs,
        )

        if type(avgpool_size) != int:
            self.avgpool = nn.AdaptiveAvgPool2d(avgpool_size)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))

        self.classifier = self.make_classifier(
            avgpool_size,
            nodes=fc_nodes,
            num_classes=num_classes,
            out_features=out_channels,
        )

    def forward(self, inp: torch.tensor) -> torch.tensor:
        out = self.features(inp)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def make_block(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        pad: str = "reflect",
        bias: bool = False,
        normalisation: str = "batch",
        activation: str = "relu",
        initialisation: str = "kaiming",
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {},
    ) -> nn.Sequential:

        layers = []
        layers.append(
            ConvLayer(
                in_channels,
                out_channels,
                kernel=kernel,
                stride=stride,
                pad=pad,
                bias=bias,
                normalisation=normalisation,
                activation=activation,
                initialisation=initialisation,
                conv_kwargs=conv_kwargs,
                norm_kwargs=norm_kwargs,
                act_kwargs=act_kwargs,
            )
        )

        for _ in range(1, num_layers):
            layers.append(
                ConvLayer(
                    out_channels,
                    out_channels,
                    kernel=kernel,
                    stride=1,
                    pad=pad,
                    bias=bias,
                    normalisation=normalisation,
                    activation=activation,
                    initialisation=initialisation,
                    conv_kwargs=conv_kwargs,
                    norm_kwargs=norm_kwargs,
                    act_kwargs=act_kwargs,
                )
            )

        return nn.Sequential(*layers)

    def make_feature_extractor(
        self,
        layers: Sequence[int],
        in_channels: int,
        out_channels: int,
        intermediate_channels: Sequence[int],
        kernel_sizes: Union[int, Sequence[int], Sequence[Sequence[int]]],
        strides: Union[int, Sequence[int], Sequence[Sequence[int]]],
        pad: str = "reflect",
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {},
    ) -> nn.Sequential:

        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes for _ in range(len(layers))]
        if type(strides) == int:
            strides = [strides for _ in range(len(layers))]

        blocks = []
        for b in range(len(layers)):
            if b == 0:
                blocks.append(
                    self.make_block(
                        layers[b],
                        in_channels,
                        intermediate_channels[b],
                        kernel=kernel_sizes[b],
                        stride=strides[b],
                        pad=pad,
                        bias=bias,
                        normalisation=normalisation,
                        activation=activation,
                        initialisation=initialisation,
                        conv_kwargs=conv_kwargs,
                        norm_kwargs=norm_kwargs,
                        act_kwargs=act_kwargs,
                    )
                )
            elif b == len(layers) - 1:
                blocks.append(
                    self.make_block(
                        layers[b],
                        intermediate_channels[b],
                        out_channels,
                        kernel=kernel_sizes[b],
                        stride=strides[b],
                        pad=pad,
                        bias=bias,
                        normalisation=normalisation,
                        activation=activation,
                        initialisation=initialisation,
                        conv_kwargs=conv_kwargs,
                        norm_kwargs=norm_kwargs,
                        act_kwargs=act_kwargs,
                    )
                )
            else:
                blocks.append(
                    self.make_block(
                        layers[b],
                        intermediate_channels[b],
                        intermediate_channels[b + 1],
                        kernel=kernel_sizes[b],
                        stride=strides[b],
                        pad=pad,
                        bias=bias,
                        normalisation=normalisation,
                        activation=activation,
                        initialisation=initialisation,
                        conv_kwargs=conv_kwargs,
                        norm_kwargs=norm_kwargs,
                        act_kwargs=act_kwargs,
                    )
                )

        return nn.Sequential(*blocks)

    def make_classifier(
        self,
        avgpool_size: Union[int, Sequence[int]],
        nodes: int,
        num_classes: int,
        out_features: int,
    ) -> nn.Sequential:

        if type(avgpool_size) != int:
            y, x = avgpool_size
        else:
            y, x = avgpool_size, avgpool_size

        return nn.Sequential(
            nn.Linear(out_features * y * x, nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(nodes, nodes),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(nodes, num_classes),
        )


class ResNet(nn.Module):
    """
    This is a class for building a custom Residual Network (ResNet) for classification.
    """

    def __init__(
        self,
        layers: Sequence[int],
        num_classes: int,
        in_channels: int,
        out_channels: int,
        intermediate_channels: Sequence[int],
        kernel_sizes: Union[int, Sequence[int], Sequence[Sequence[int]]],
        strides: Union[int, Sequence[int], Sequence[Sequence[int]]],
        pad: str = "reflect",
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {},
        avgpool_size: Union[int, Sequence[int]] = 7,
    ) -> None:
        super().__init__()

        self.features = self.make_feature_extractor(
            layers,
            in_channels,
            out_channels,
            intermediate_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pad=pad,
            bias=bias,
            normalisation=normalisation,
            activation=activation,
            initialisation=initialisation,
            conv_kwargs=conv_kwargs,
            norm_kwargs=norm_kwargs,
            act_kwargs=act_kwargs,
        )

        if type(avgpool_size) != int:
            self.avgpool = nn.AdaptiveAvgPool2d(avgpool_size)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))

        self.classifier = self.make_classifier(
            avgpool_size=avgpool_size,
            num_classes=num_classes,
            out_features=out_channels,
        )

    def make_block(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        pad: str = "reflect",
        bias: bool = False,
        normalisation: str = "batch",
        activation: str = "relu",
        initialisation: str = "kaiming",
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {},
    ) -> nn.Sequential:

        layers = []
        layers.append(
            ResLayer(
                in_channels,
                out_channels,
                kernel=kernel,
                stride=stride,
                pad=pad,
                bias=bias,
                normalisation=normalisation,
                activation=activation,
                initialisation=initialisation,
                conv_kwargs=conv_kwargs,
                norm_kwargs=norm_kwargs,
                act_kwargs=act_kwargs,
            )
        )

        for _ in range(1, num_layers):
            layers.append(
                ResLayer(
                    out_channels,
                    out_channels,
                    kernel=kernel,
                    stride=1,
                    pad=pad,
                    bias=bias,
                    normalisation=normalisation,
                    activation=activation,
                    initialisation=initialisation,
                    conv_kwargs=conv_kwargs,
                    norm_kwargs=norm_kwargs,
                    act_kwargs=act_kwargs,
                )
            )

        return nn.Sequential(*layers)

    def make_feature_extractor(
        self,
        layers: Sequence[int],
        in_channels: int,
        out_channels: int,
        intermediate_channels: Sequence[int],
        kernel_sizes: Union[int, Sequence[int], Sequence[Sequence[int]]],
        strides: Union[int, Sequence[int], Sequence[Sequence[int]]],
        pad: str = "reflect",
        bias: bool = False,
        normalisation: Optional[str] = None,
        activation: str = "relu",
        initialisation: str = "kaiming",
        conv_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        act_kwargs: Dict = {},
    ) -> nn.Sequential:

        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes for _ in range(len(layers))]
        if type(strides) == int:
            strides = [strides for _ in range(len(layers))]

        conv_kernel_size = (
            2 * kernel_sizes + 1
            if type(kernel_sizes) == int
            else 2 * kernel_sizes[0] + 1
        )

        blocks = []
        blocks.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=intermediate_channels[0],
                kernel=conv_kernel_size,
                stride=strides[0],
                pad=pad,
                bias=bias,
                normalisation=normalisation,
                activation=activation,
                initialisation=initialisation,
            )
        )
        for b in range(len(layers)):
            if b == 0:
                blocks.append(
                    self.make_block(
                        layers[b],
                        intermediate_channels[b],
                        intermediate_channels[b],
                        kernel=kernel_sizes[b],
                        stride=1,
                        pad=pad,
                        bias=bias,
                        normalisation=normalisation,
                        activation=activation,
                        initialisation=initialisation,
                        conv_kwargs=conv_kwargs,
                        norm_kwargs=norm_kwargs,
                        act_kwargs=act_kwargs,
                    )
                )
            elif b == len(layers) - 1:
                blocks.append(
                    self.make_block(
                        layers[b],
                        intermediate_channels[b],
                        out_channels,
                        kernel=kernel_sizes[b],
                        stride=strides[b],
                        pad=pad,
                        bias=bias,
                        normalisation=normalisation,
                        activation=activation,
                        initialisation=initialisation,
                        conv_kwargs=conv_kwargs,
                        norm_kwargs=norm_kwargs,
                        act_kwargs=act_kwargs,
                    )
                )
            else:
                blocks.append(
                    self.make_block(
                        layers[b],
                        intermediate_channels[b],
                        intermediate_channels[b + 1],
                        kernel=kernel_sizes[b],
                        stride=strides[b],
                        pad=pad,
                        bias=bias,
                        normalisation=normalisation,
                        activation=activation,
                        initialisation=initialisation,
                        conv_kwargs=conv_kwargs,
                        norm_kwargs=norm_kwargs,
                        act_kwargs=act_kwargs,
                    )
                )

        return nn.Sequential(*blocks)

    def make_classifier(
        self,
        avgpool_size: Union[int, Sequence[int]],
        num_classes: int,
        out_features: int,
    ) -> nn.Linear:

        if type(avgpool_size) != int:
            y, x = avgpool_size
        else:
            y, x = avgpool_size, avgpool_size

        return nn.Linear(out_features * y * x, num_classes)
