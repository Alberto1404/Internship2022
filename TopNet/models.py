import sys
import os
import numpy as np
import torch
from torch.nn import SmoothL1Loss, MSELoss, ReLU, Module
from monai.losses import DiceCELoss

from unetr_topnet_2dec import UNETR_topnet_2
from unetr_topnet_2dec_ori import UNETR_topnet_2_ori
from unetr_topnet_3dec import UNETR_topnet_3

# Add topology functions
sys.path.insert(0,'...') # Modify clDice topology path
from clDice.cldice_loss.cldice import soft_dice, soft_dice_cldice

class C_loss (Module):

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


def get_model_loss(args):

	model = UNETR_topnet_2(
			in_channels=1,
			out_channels=1,
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

	
	loss_function_1 = DiceCELoss(sigmoid=True, to_onehot_y=False)
	loss_function_2 = C_loss()
	loss_function_3 = MSELoss()

	"""if len(args.decoder) == 2:
		return model, [loss_function_1, loss_function_2, loss_function_3]
	else:
		if args.decoder[0] == 'dmap':
			return model, [loss_function_1, loss_function_2]
		else:
			return model, [loss_function_1, loss_function_3]"""
	return model, [loss_function_1, loss_function_2]
		
