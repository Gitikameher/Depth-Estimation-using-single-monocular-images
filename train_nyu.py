import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import transforms
from dataset import NYUDataset
from custom_transforms import *

import plot_utils
from model_utils import *
from plot_utils import *
from nn_model import Net

#3x640x480 in dataset,   CxWxH
#480x640x3 for plotting, HxWxC
#3x480x640 for pytorch,  CxHxW  

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

n_epochs = 20
lr = 2e-5
bs = 8
sz = (320,240)
seed = np.random.seed(1)
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
mean, std = torch.tensor(mean), torch.tensor(std)
unnormalize = UnNormalizeImgBatch(mean, std)
tfms = transforms.Compose([
    ResizeImgAndDepth(sz),
    RandomHorizontalFlip(),
    ImgAndDepthToTensor(),
    NormalizeImg(mean, std)
])
ds = NYUDataset('data/', tfms)
dl = torch.utils.data.DataLoader(ds, bs, shuffle=True)
train_loader, val_loader, test_loader = ds.create_split_loaders(bs, seed, tfms,0.1, 0.1,True)

# i = 1
# plot_utils.plot_image(get_unnormalized_ds_item(unnormalize, ds[i]))

model = Net()
model.to(device)
output_dir = "nyu"
make_dir(output_dir)
images_dir = os.path.join(output_dir,"saved_images")
make_dir(images_dir)
epoch_tracker = EpochTracker(os.path.join(output_dir, "nyu_epoch.txt"))

if epoch_tracker.epoch > 0:
    start_epoch = epoch_tracker.epoch + 1
    model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint_%d.pth.tar' % epoch_tracker.epoch)))
    losses_logger = open(os.path.join(output_dir, 'losses_log.txt'), 'a')
else:
    start_epoch = 0
    losses_logger = open(os.path.join(output_dir, 'losses_log.txt'), 'w')
    
depth_loss = ScaleInvariantLoss(lamada=1.0)

def validate(model):
    final_loss_val=[]
    final_mse_loss_val=[]
    final_l1_loss_val=[]
    final_berhu_loss_val=[]
    
    model.eval()
    
    with torch.no_grad():
        for batch_val, labels in val_loader:   
            batch = batch_val.to(device)
            labels = labels.to(device).unsqueeze(1)

            preds = model(batch)
            preds = (preds * 0.225) + 0.45
            loss = depth_loss(preds, labels) 
            mse_loss = MaskedMSELoss(preds, labels)
            l1_loss = MaskedL1Loss(preds, labels)
            berhu_loss = berHuLoss(preds, labels)

            final_loss_val.append(loss.item())
            final_mse_loss_val.append(mse_loss.item())
            final_l1_loss_val.append(l1_loss.item())
            final_berhu_loss_val.append(berhu_loss.item())
    model.train()
    return (np.mean(final_loss_val), np.mean(final_mse_loss_val), 
            np.mean(final_l1_loss_val), np.mean(final_berhu_loss_val))

model.train()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)

total_steps = 0
total_batches= len(train_loader)

for e in range(start_epoch, n_epochs):
    final_loss=[]
    final_mse_loss=[]
    final_l1_loss=[]
    final_berhu_loss=[]
    for i, data_train in enumerate(train_loader):
        batch, labels = data_train
        
        optimizer.zero_grad()
        
        batch = batch.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        preds = model(batch)
        preds = (preds * 0.225) + 0.45
        loss = depth_loss(preds, labels) 
        final_loss.append(loss.item())
        
#         print(preds.shape, labels.shape)
        mse_loss = MaskedMSELoss(preds, labels)
        final_mse_loss.append(mse_loss.item())
        
        l1_loss = MaskedL1Loss(preds, labels)
        final_l1_loss.append(l1_loss.item())
        
        berhu_loss = berHuLoss(preds, labels)
        final_berhu_loss.append(berhu_loss.item())
        
        loss.backward()
        optimizer.step()
        
        total_steps +=1
        if (i+1) % 5 ==0:
            res = "E:{} B:({}/{}) Loss:{:0.3f} L1:{:0.3f} MSE:{:0.3f} berHu:{:0.3f}".format(e, i+1, total_batches, 
                                                                       loss.item(), l1_loss.item(),
                                                                       mse_loss.item(), berhu_loss.item())
            print(res)
        if (i+1) % 20 == 0:
            pred = preds[0].detach()
            img_new = unnormalize(batch[0].cpu()).detach().squeeze()
            depth=labels[0].detach()
            
            save_image(pred, img_new, depth, i, e, output_dir)
        
    del batch
    del labels
    val_depth, val_mse, val_l1, val_berhu = validate(model)
    train_depth = np.mean(final_loss)
    train_mse = np.mean(final_mse_loss)
    train_l1 = np.mean(final_l1_loss)
    train_berhu = np.mean(final_berhu_loss)
    result = "{} {} {} {} {} {} {} {} {}\n".format(e+1, train_depth, val_depth, 
                                                 train_l1, val_l1, train_mse, val_mse,
                                                 train_berhu, val_berhu)
    losses_logger.write(result)
    losses_logger.flush()
    epoch_tracker.write(e)
    torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_%d.pth.tar"%e))                                                  