import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skimage.morphology import binary_dilation, skeletonize_3d
from skimage.util import img_as_float32

from cldice_loss.soft_skeleton import soft_skel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class soft_cldice(nn.Module):
    def __init__(self, iter_=16, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        iters_ = self.iter
        smooth = self.smooth

        skel_pred = soft_skel(y_pred, iters_)
        skel_true = soft_skel(y_true, iters_)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+smooth)/(torch.sum(skel_pred)+smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+smooth)/(torch.sum(skel_true)+smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred, smooth=1):
    """[function to compute dice loss]
    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]
    Returns:
        [float32]: [loss value]
    """
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)




class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=16, alpha=0.3, smooth = 1):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true):
        
        y_pred = self.sigmoid(y_pred)

        y_true_skel = torch.zeros_like(y_true).to(device)
        y_pred_skel = torch.zeros_like(y_pred).to(device)
        for idx, (vol_true, vol_pred) in enumerate(zip(y_true, y_pred)): # Iterate in batch size
            y_true_skel[idx,0,:,:,:] = torch.from_numpy(binary_dilation(binary_dilation(img_as_float32(skeletonize_3d(vol_true.squeeze().cpu().numpy()))).astype(np.float32)).astype(np.float32)).to(device)
            y_pred_skel[idx, 0,:,:,:] = torch.from_numpy(binary_dilation(binary_dilation(img_as_float32(skeletonize_3d(vol_pred.squeeze().cpu().detach().numpy()))).astype(np.float32)).astype(np.float32)).to(device)
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred_skel, self.iter) # y_pred BEFORE
        skel_true = soft_skel(y_true_skel, self.iter) # y_true BEFORE
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)

        return (1.0-self.alpha)*dice+self.alpha*cl_dice
        


"""class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=16, alpha=0.5, smooth = 1., binary = True):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.binary = binary
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_true, y_pred):
        if self.binary:
            y_pred = self.sigmoid(y_pred)
        else:
            y_pred = self.softmax(y_pred)
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice"""

