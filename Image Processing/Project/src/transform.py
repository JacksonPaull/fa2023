# ====================== imports =====================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import logging
import time
from tqdm import tqdm

# =========================== Parameters =========================

IMG_SIZE_BIG = 1080
IMG_SIZE_SMALL = 128


# ========== Initialization and Global Variables ==================
logger = logging.getLogger('styleTransfer')

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
torch.set_default_device(device)
logger.log(logging.INFO, f'Torch device set to {dev}')

imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

# ====================================================================

def load_image(image_name):
    global loader
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0) # fake batch dimension required to fit network's input dimensions
    return image.to(device, torch.float)


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    
class TwoStyleLoss(nn.Module):
    def __init__(self, target_feature1, target_feature2, lamb):
        super(TwoStyleLoss, self).__init__()
        self.target1 = gram_matrix(target_feature1).detach()
        self.target2 = gram_matrix(target_feature2).detach()
        self.lamb = lamb

    def forward(self, input):
        G = gram_matrix(input)
        self.loss1 = F.mse_loss(G, self.target1)
        self.loss2 = F.mse_loss(G, self.target2)

        self.loss = self.lamb * self.loss1 + (1-self.lamb) * self.loss2
        return input
    
# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

        if torch.cuda.is_available():
            self.mean.to('cuda:0')
            self.std.to('cuda:0')

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std
    

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, lamb,
                               content_layers=None,
                               style_layers=None, 
                               style_img_2=None, 
                               use_two_styles=False):
    if content_layers is None:
        content_layers = ['conv_4']

    if style_layers is None:
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            if use_two_styles:
                target_feature2 = model(style_img_2).detach()
                style_loss = TwoStyleLoss(target_feature, target_feature2, lamb)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
            else:
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) or isinstance(model[i], TwoStyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, style_img_2=None, lamb=0.5, input_img=None, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    start = time.time()
    logger.log(logging.DEBUG, 'Building the style transfer model..')

    if input_img is None:
        input_img = content_img.clone()

    use_two_style = style_img_2 is not None

    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, 
        style_img, content_img, 
        style_img_2=style_img_2, use_two_styles=use_two_style, lamb=lamb)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    logger.log(logging.DEBUG, 'Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                logger.log(logging.DEBUG, "run {}:".format(run))
                logger.log(logging.DEBUG, 'Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)
    end = time.time()
    logger.info(f'Finished creating image in {end - start:.2f}s ({(end-start)/60:.2f}min)')
    return input_img

unloader = transforms.ToPILImage() 
def to_PIL(tensor):
    global unloader
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    return image

def reconstruct_style_img(cnn, normalization_mean, normalization_std, 
                          style_img, num_steps=300, style_layers=None):
    input_img = torch.rand(style_img.shape, dtype=style_img.dtype)

    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, 
        style_img, content_img=None, lamb=None,
        content_layers=[])  


    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)

    for run in tqdm(range(num_steps)):

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            loss = 0

            for sl in style_losses:
                loss += sl.loss

            loss.backward()
            return loss

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img
