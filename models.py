from monai.networks.nets import UNETR, UNet
from monai.losses import DiceCELoss

import sys
import os

from torch import sigmoid

# Add topology functions
sys.path.insert(0,os.path.join('/home2/alberto/code', 'clDice')) # Modify topology path
from clDice.cldice_loss.cldice import soft_dice, soft_dice_cldice

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
		model = UNETR(
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
			dropout_rate = args.dropout_rate
		)

	# loss_function = soft_dice_cldice() if args.cldice == True else (DiceCELoss(sigmoid=True, to_onehot_y=False) if args.binary == True else DiceCELoss(softmax=True, to_onehot_y=True))
	if args.metric == 'softdice':
		loss_function = soft_dice_cldice(binary=True) if args.binary else soft_dice_cldice(binary=False)
	else:
		if args.binary:
			loss_function = DiceCELoss(sigmoid=True, to_onehot_y=False)
		else:
			loss_function = DiceCELoss(softmax=True, to_onehot_y=True)

	return model, loss_function