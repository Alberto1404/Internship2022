### In models.py appear definitions of: 
# - C_loss and L2_loss (loss functions for D2 and D3 decoders respectively)
# - Selection of the architecture to use for training: 
# 	· UNETR_topnet_2: (D1+D2)
# 	· UNETR_topnet_2_ori: (D1+D3)
# 	· UNETR_topnet_3: (D1+D2+D3) (NOT USED)

from unetr_topnet_2dec import UNETR_topnet_2
from unetr_topnet_2dec_ori import UNETR_topnet_2_ori
from unetr_topnet_3dec import UNETR_topnet_3
from monai.losses import DiceCELoss

import sys
import os
import numpy as np
import torch
from torch.nn import SmoothL1Loss, MSELoss, ReLU, Module

# Add topology functions
# sys.path.insert(0,os.path.join('/home2/alberto/aux_TOPNET/code_TopNet', 'clDice')) # Modify topology path
# from clDice.cldice_loss.cldice import soft_dice, soft_dice_cldice

class C_loss (Module): # D2

	def __init__(self, beta = 1):
		super(C_loss, self).__init__()
		self.beta = beta
		self.smoothl1 = SmoothL1Loss(beta = self.beta, reduction='none')
		self.relu = ReLU(inplace = True)

	def forward(self, y_pred, y_true, vessel):
		y_pred = self.relu(y_pred)

		C_loss = torch.multiply(
								(1/torch.pow(y_true,2)).where( vessel > 0.5, torch.tensor(0.0).to('cuda') ),
								self.smoothl1(y_pred, y_true)
								)
		C_loss = torch.sum(C_loss) / torch.count_nonzero(vessel)

		return C_loss

class L2_loss(Module):
	def __init__(self):
		super(L2_loss, self).__init__()
		self.mse = MSELoss()
		self.relu = ReLU(inplace=True)

	def forward(self, y_pred, y_true):
		y_pred = self.relu(y_pred)
		L2 = self.mse(y_pred, y_true)
		return L2

def get_model_loss(args):

	if len(args.decoder) == 2:
		model = UNETR_topnet_3(
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
	else:
		if args.decoder[0] == 'dmap':
			model = UNETR_topnet_2(
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
		else:
			model = UNETR_topnet_2_ori(
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

	
	# loss_function = soft_dice_cldice() if args.cldice == True else (DiceCELoss(sigmoid=True, to_onehot_y=False) if args.binary == True else DiceCELoss(softmax=True, to_onehot_y=True)

	
	if args.binary:
		loss_function_1 = DiceCELoss(sigmoid=True, to_onehot_y=False)
	else:
		loss_function_1 = DiceCELoss(softmax=True, to_onehot_y=True)
	loss_function_2 = C_loss()
	# loss_function_3 = L1Loss()
	loss_function_3 = L2_loss() # MSELoss()

	if len(args.decoder) == 2:
		return model, [loss_function_1, loss_function_2, loss_function_3]
	else:
		if args.decoder[0] == 'dmap':
			return model, [loss_function_1, loss_function_2]
		else:
			return model, [loss_function_1, loss_function_3]
		
