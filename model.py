from email.mime import image
from unittest.mock import patch
import torch
import torch.nn as nn
from vit_pytorch.vit_for_small_dataset import ViT
from dimensionality_reduction import AE
from typing import Dict

class MVCNN(nn.Module):
    """
    The base class for the multiview CNN for learning image classification based on different views of a larger field of view.
    """

    def __init__(self, cnn: nn.Module, num_classes: int) -> None:
        super(MVCNN, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")
        self.features = cnn.features
        # self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = cnn.classifier

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.transpose(0,1) #switches the view dimension with the batch dimension
        # print(x.shape)

        view_pool = [] #create an empty list to store all features for each view

        for j, v in enumerate(x):
            # print(j)
            v = self.features(v)
            # print(v.shape)
            #v will have shape (batch_size, C, H, W)
            #want to reshape it for the fully-connected layers as (batch_size, -1), the -1 combines all other dimensions
            # v = self.avgpool(v)
            v = v.view(v.size(0), -1)
            # print(v.shape)

            view_pool.append(v)
            # if j == 0:
            #     pooled_view = v
            # else:
            #     pooled_view, _ = torch.max(torch.stack([pooled_view,v]), dim=0)
            # del(v)

        pooled_view, _ = torch.max(torch.stack(view_pool), dim=0) #inline with the literature, the elementwise maximum is taken across the view dimension to complete the viewpooling
        #here torch.max() returns the elementwise maximum along the view dimension but will also return which dimension each corresponding maximum came from hence the _

        # pooled_view = pooled_view.to(self.device)

        pooled_view = self.classifier(pooled_view)

        return pooled_view

class AEViT(nn.Module):
    """
    Vision Transformer model with autoencoder input.
    """

    def __init__(self, in_channels: int, ae_hidden: int, ae_model: str, image_size: int, patch_size: int, num_classes: int, vit_dim: int, vit_depth: int, vit_heads: int, vit_mlp_dim: int, y_patches: int, x_patches: int, vit_kwargs: Dict = {}) -> None:
        super().__init__()

        self.ae = AE(in_channels=in_channels, nef=ae_hidden)
        self.ae.load_state_dict(torch.load(ae_model)["model_state_dict"])
        self.ae = self.ae.encoder
        for param in self.ae.parameters():
            param.requires_grad = False

        assert (image_size == patch_size * y_patches or patch_size * x_patches) 
        self.vit = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=vit_dim, depth=vit_depth, heads=vit_heads, mlp_dim=vit_mlp_dim, channels=in_channels, **vit_kwargs)

        self.y_patches = y_patches
        self.x_patches = x_patches

    def forward(self, x: torch.tensor) -> torch.tensor:
        b, p, c, h1, w1 = x.shape
        x = x.reshape(b*p, c, h1, w1)
        x = self.ae(x)
        _, _, h2, w2 = x.shape
        x = x.reshape(b, p, c, h2, w2)
        x = x.reshape(b, c, self.y_patches*h2, self.x_patches*w2)

        out = self.vit(x)
        return out