from unetr_topnet import UNETR_topnet
from monai.losses import DiceCELoss

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import SmoothL1Loss

# Add topology functions
sys.path.insert(0,os.path.join(os.path.abspath(os.getcwd()), 'clDice')) # Modify topology path
from clDice.cldice_loss.cldice import soft_dice, soft_dice_cldice

class C_loss(nn.Module):

	def __init__(self, beta = 1):
		super(C_loss, self).__init__()
		self.beta = beta
		self.smoothl1 = nn.SmoothL1Loss(beta = self.beta, reduction='none')
		self.relu = nn.ReLU(inplace = True)

	def forward(self, y_pred, y_true, vessel):
		y_pred = self.relu(y_pred)
		
		C_loss = torch.multiply(1/torch.pow(y_true,2),self.smoothl1(self.relu(y_pred), y_true)).where( vessel == 1, torch.tensor(0.0).to("cuda" if torch.cuda.is_available() else "cpu"))
		C_loss = torch.sum(C_loss) / torch.count_nonzero(vessel)
		return C_loss
		
		
		return torch.sum(torch.multiply((1/torch.pow(y_true,2)).where( vessel > 0.5, torch.tensor(0.0).to('cuda')), self.smoothl1(y_pred, y_true))) / torch.count_nonzero(vessel)

def get_model_loss(args):


	if args.net == 'unet':
		model = UNet(
			spatial_dims=3,
			in_channels=1,
			out_channels= 1 if args.binary == True else 3,
			channels=(16, 32, 64, 128, 256),
			strides=(2, 2, 2, 2),
			num_res_units=2,
		)
		
	else:
		model = UNETR_topnet(
			in_channels=1,
			out_channels=1 if args.binary == True else 3,
			img_size = args.input_size,
			feature_size = args.feature_size,
			hidden_size = args.hidden_size, 
			mlp_dim = args.mlp_dim, 
			num_heads = args.num_heads,
			pos_embed = args.pos_embed,
			norm_name = args.norm_name,
			res_block = args.res_block,
			dropout_rate = args.dropout_rate,
			spatial_dims=3
		)

	# loss_function = soft_dice_cldice() if args.cldice == True else (DiceCELoss(sigmoid=True, to_onehot_y=False) if args.binary == True else DiceCELoss(softmax=True, to_onehot_y=True))
	if args.metric == 'softdice':
		loss_function = soft_dice_cldice(binary=True) if args.binary else soft_dice_cldice(binary=False)
	else:
		if args.binary:
			loss_function_1 = DiceCELoss(sigmoid=True, to_onehot_y=False)
			loss_function_2 = C_loss()
		else:
			loss_function_1 = DiceCELoss(softmax=True, to_onehot_y=True)
			loss_function_2 = C_loss()

	return model, loss_function_1, loss_function_2
