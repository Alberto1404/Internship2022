import soft_skeleton
import nibabel as nib
import numpy as np
import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt

import soft_skeleton

from skimage.morphology import skeletonize_3d
from skimage.util import img_as_float32
from scipy.ndimage.morphology import distance_transform_edt as DTM
from sklearn.cluster import KMeans

def reshape_matrix(matrix):
    reshaped = np.zeros((matrix.shape[-1], matrix.shape[0]*matrix.shape[1]))
    for slice in range(matrix.shape[-1]):
        reshaped[slice,:] = matrix[:,:,slice].flatten()

    return reshaped.T

def display_KFCV(metrics_all): # CADA FOLD ES EL MAXLINE MINLINE DEL EJEMPLO. TU AHORA PON EN LABELS TRAINING Y VALIDATION

    x = np.arange(metrics_all.shape[-1]) + 1
    """fig = plt.figure("train", (12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = ax1.twinx()"""
    mean_train_loss = np.mean(metrics_all[:,0,:], axis = 0)
    mean_val_loss = np.mean(metrics_all[:,1,:], axis = 0)
    mean_val_metric = np.mean(metrics_all[:,2,:], axis = 0)

    for fold_idx, fold in enumerate(metrics_all):
        loss_training = fold[0,:]
        loss_validation = fold[1,:]
        dice_validation = fold[2,:]

        x = np.arange(metrics_all.shape[-1]) + 1
        fig = plt.figure(fold_idx, (12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = ax1.twinx()
        
        ax1.plot(x, loss_training, label = 'Training loss')
        ax1.plot(x, loss_validation, label = 'Validation loss')
        ax1.plot(x, mean_train_loss, '-,',label = 'KFCV Training Loss')
        ax1.plot(x, mean_val_loss, '-,', label = 'KFCV Validation Loss')
        
        ax2.plot(x, dice_validation, label = 'Validation metric')
        ax2.plot(x, mean_val_metric, '-,', label = 'KFCV Validation Metric')
        
        ax1.legend(loc = 'center right', prop={'size': 6})
        ax2.legend(prop={'size': 6})

    plt.show()
    # leg2 = ax1.legend([l[0] for l in plot_losses], ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5'])
    # ax1.clf().add_artist(leg1)

def main():
    metrics_all = loadmat('/home/guijosa/Documents/PythonDocs/metrics_all.mat')
    metrics = metrics_all['metrics_all']
    display_KFCV(metrics)


if __name__ == '__main__':

    main()
