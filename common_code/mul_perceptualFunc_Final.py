#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


# VGGnet only
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        self.stop_at = len(extracted_layers)

    def forward(self, x):
        outputs = []
        count = 0
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                count += 1
                outputs.append(x)
                if count == self.stop_at:
                    return outputs
        return outputs


# default 2
def create_perceptual_loss2(value, img_tensor, model, gamma=10000,
                            scalar=1, layers=['9', '19', '29', '38', '48']):
    img_tensor = img_tensor.detach()
    featex = FeatureExtractor(model.features, layers)
    target = model(img_tensor).argmax().detach()
    init_feat = featex(img_tensor)
    L2 = torch.nn.MSELoss()

    def loss(img):
        # perceptual loss
        out = 0
        for f, g in zip(init_feat, featex(img)):
            out += L2(f, g)
        out *= gamma
        out += L2(img, img_tensor)*scalar

        # l2 loss of response
        response = model(img)
        others = torch.max(response[0, :target].max(),
                           response[0, target + 1:].max())
        lab = response[0, target]
        return out + (lab - others - value)**2
    return loss


# multi-class chris
def create_perceptual_loss_multiclass(value, img_tensor, model, gamma=10000,
                                      scalar=1,
                                      layers=['9', '19', '29', '38', '48'],
                                      targetID=None):
    img_tensor = img_tensor.detach()
    featex = FeatureExtractor(model.features, layers)
    target = targetID
    init_feat = featex(img_tensor)
    L2 = torch.nn.MSELoss()
    r = model(img_tensor)

    def loss(img):
        # perceptual loss
        out = 0
        for f, g in zip(init_feat, featex(img)):
            out += L2(f, g)
        out *= gamma
        out += L2(img, img_tensor)*scalar
        response = model(img)
        lab = response[0, target].max()
        return out + (lab - value)**2
    return loss


def find_direction(
        loss,
        factual,
        mins=False,
        maxs=False,
        iterations=200
        ):

    direction = factual.data.clone()
    direction.requires_grad = True
    # set up loss
    optimizer = torch.optim.LBFGS([direction, ], max_iter=iterations)

    def closure():
        # Clamp_ doesn't take vectors
        # This can constrain the output if desired
        if mins is not False:
            min_mask = direction.data < mins
            direction.data[min_mask] = mins[min_mask]
        if maxs is not False:
            max_mask = direction.data > maxs
            direction.data[max_mask] = mins[max_mask]
        l = loss(direction)
        optimizer.zero_grad()
        l.backward(retain_graph=True)
        return l
    optimizer.step(closure)

    return direction


def diff_renorm(image):
    """Maps image back into [0,1]. Useful for visualising differences"""
    scale = 0.5/image.abs().max()
    image = image*scale
    image += 0.5
    return image


def vis(im, text=False):
    plt.figure()
    if text:
        plt.title(text.split(', ')[0])
    if im.detach().cpu().numpy().ndim == 4:
        plt.imshow(im.detach().cpu().numpy()[0].transpose(1, 2, 0))
    else:
        plt.imshow(im.detach().cpu().numpy()[0])
        plt.colorbar()
    plt.axis('off')
    if text is not False:
        plt.savefig(
            text.split(', ')[0].replace(
                ' ', '').replace('/', ''), bbox_inches='tight')
