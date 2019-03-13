import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
from matplotlib import pyplot as plt
from skimage.transform import resize

cmap = plt.cm.jet

def plot_image(tup):
    img_tensor, depth_tensor = tup
    fig, axes = plt.subplots(1, 2, figsize=(10,15))
    for i,ax in enumerate(axes.flat):
        if(i==0):
            plot_image_tensor_in_subplot(ax, img_tensor)
        else:
            plot_depth_tensor_in_subplot(ax, depth_tensor)
        hide_subplot_axes(ax)

    plt.tight_layout()
    
#subplot utils    
def hide_subplot_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_image_tensor_in_subplot(ax, img_tensor):
    im = img_tensor.cpu().numpy().transpose((1,2,0))
    #pil_im = Image.fromarray(im, 'RGB')
    ax.imshow(im)

def plot_depth_tensor_in_subplot(ax, depth_tensor):
    im = depth_tensor.cpu().numpy()
    im = im*255
    im = im.astype(np.uint8)
#     #pil_im = Image.fromarray(im, 'L')
    ax.imshow(im,'gray')
    
def plot_model_predictions_on_sample_batch(images, depths, preds, plot_from=0, figsize=(12,12)):
    n_items=5
    fig, axes = plt.subplots(n_items, 3, figsize=figsize)
    
    for i in range(n_items):
        plot_image_tensor_in_subplot(axes[i,0], images[plot_from+i])
        plot_depth_tensor_in_subplot(axes[i,1], depths[plot_from+i])
        plot_depth_tensor_in_subplot(axes[i,2], preds[plot_from+i])
        hide_subplot_axes(axes[i,0])
        hide_subplot_axes(axes[i,1])
        hide_subplot_axes(axes[i,2])
    
    plt.tight_layout()

def create_depth_color(depth):
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth = (255 * cmap(depth_relative)[:, :, :3])
    return depth

def save_image(pred_depth, x, y, batch, epoch, mode="train"):
    depth = create_depth_color(np.transpose(pred_depth.cpu().numpy(), [1,2,0])[:, :, 0])
    target = create_depth_color(np.transpose(y.cpu().numpy(), [1,2,0])[:, :, 0])
    orig = 255 * x.cpu().numpy()
    orig = resize(np.transpose(orig, (1, 2, 0)), (120, 160))
    img = np.concatenate((orig, target, depth), axis =1)

    img = Image.fromarray(img.astype('uint8'))
    img.save('saved_images/%s_image_%d_%d.jpg'%(mode, epoch, batch))