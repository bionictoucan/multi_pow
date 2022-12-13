import torch.nn as nn
import torchvision.models as models

def vgg11(model_type: str) -> models.vgg.VGG:
    if model_type == "binary":
        model = models.vgg11_bn(weights="VGG11_BN_Weights.IMAGENET1K_V1") #model is initialised using ImageNet weights
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1)
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
            nn.Linear(4096, 3)
        )
        nn.init.kaiming_normal_(model.features[0].weight, nonlinearity="relu")
        nn.init.zeros_(model.features[0].bias)
        for m in model.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    return model