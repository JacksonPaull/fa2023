import torch
import torchvision.models

vgg19_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
vgg19_normalization_std = torch.tensor([0.229, 0.224, 0.225])
vgg_19_default_weights = torchvision.models.VGG19_Weights.DEFAULT
vgg19 = torchvision.models.vgg19