
# Loading libraries

# Image libraries and image processing functions
import torchvision
import pickle
from scipy.ndimage import gaussian_filter

# Our method code
from common_code.mul_perceptualFunc_Final import *
from common_code import utils

# Plotting libraries
import matplotlib.pyplot as plt

image_name = 'ILSVRC2012_val_00000051.JPEG'

# load the image
img_tensor = utils.open_and_preprocess(image_name)
img_variable = img_tensor.unsqueeze_(0)

#  Load model
#  If you wish to use a non vgg-like model a
#  modification may be required to the
#  feature extractor which
model = torchvision.models.vgg19_bn(pretrained=True)
model.requires_grad = False
model.eval()


# List of layers to regularise
# They should correspond to ReLU layers in the network
layerLists = ['16', '19', '22', '25', '29', '32']

# create the perceptual loss with the required parameters
loss = create_perceptual_loss2(-2, img_variable, model, gamma=10000, scalar=1,
                               layers=layerLists)

# optimise the loss to find the adv. perturbation
c = find_direction(loss, img_variable, iterations=100)

# Take pixelwise euclidean distance to get the saliency map
res = torch.sqrt(((c - img_variable)**2).mean(1))
res = res.squeeze().cpu().detach().numpy()

#  Apply gaussian blur
res = gaussian_filter(res, sigma=2)


#  Save result
f = open('result.p', 'wb')
pickle.dump(res, f)
f.close()

# Save image
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(res.squeeze())
fig.savefig('outputFig.png')

