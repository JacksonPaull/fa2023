import torch
import torchvision.models

"""Define the VGG19 Model and its normalization vectors using the torch pretrained library"""
vgg19_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
vgg19_normalization_std = torch.tensor([0.229, 0.224, 0.225])
vgg_19_default_weights = torchvision.models.VGG19_Weights.DEFAULT
vgg19 = torchvision.models.vgg19

# Note: Other models could be added here as well, but their content layers and style layers would need to be defined