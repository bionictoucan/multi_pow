import torch.nn as nn
import torchvision.models as models

def vgg11(model_type: str) -> models.vgg.VGG:
    """
    This function returns the VGG-11 model used in the paper with the correctly
    updated number of classes and the first convolutional layer replaced to one
    with one input channel since the images are greyscale. All layers are
    initialised by PyTorch's ImageNet1K weights with the custom layers
    initialised using He initialisation.

    Parameters
    ----------
    model_type : str
        Whether to return a binary model (1 output class) or a multiclass model
        (3 output classes). Options are "binary" or "multi".

    Returns
    -------
    model : torchvision.models.vgg.VGG
        The PyTorch VGG-11 model set up for the problem at hand.
    """
    if model_type == "binary":
        model = models.vgg11_bn(
            weights="VGG11_BN_Weights.IMAGENET1K_V1"
        )  # model is initialised using ImageNet weights
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1),
        )
        nn.init.kaiming_normal_(model.features[0].weight, nonlinearity="relu")
        nn.init.zeros_(model.features[0].bias)
        for m in model.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    else:
        model = models.vgg11_bn(weights="VGG11_BN_Weights.IMAGENET1K_V1")
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 3),
        )
        nn.init.kaiming_normal_(model.features[0].weight, nonlinearity="relu")
        nn.init.zeros_(model.features[0].bias)
        for m in model.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    return model
