from monai.networks.nets import UNETR, UNet
from monai.losses import DiceCELoss

def get_model(args):

	
	if args.net == 'unet':
		if args.binary == True:
			out_channels = 1
			loss_function = DiceCELoss(sigmoid=True, to_onehot_y=False)
		else:
			out_channels = 2
			loss_function = DiceCELoss(softmax=True, to_onehot_y=True)

		# out_channels = 1;loss_function = DiceCELoss(sigmoid=True, to_onehot_y=False) if args.binary == True else 2; DiceCELoss(softmax=True, to_onehot_y=True)

		model = UNet(
			spatial_dims=3,
			in_channels=1,
			out_channels= out_channels,
			channels=(16, 32, 64, 128, 256),
			strides=(2, 2, 2, 2),
			num_res_units=2,
		)
	else:
		if args.binary == True:
			out_channels = 1
			loss_function = DiceCELoss(sigmoid=True, to_onehot_y=False)
		else:
			out_channels = 2
			loss_function = DiceCELoss(softmax=True, to_onehot_y=True)
		
		# out_channels = 1;loss_function = DiceCELoss(sigmoid=True, to_onehot_y=False) if args.binary == True else 2; DiceCELoss(softmax=True, to_onehot_y=True)

		model = UNETR(
			in_channels=1,
			out_channels=out_channels,
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

	return model, loss_function