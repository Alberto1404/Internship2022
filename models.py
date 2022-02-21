from monai.networks.nets import UNETR, UNet

def get_model(model, binary_flag, img_size, feature_size, hidden_size, mlp_dim, num_heads, pos_embed, norm_name, res_block, dropout_rate):
	if model == 'unet':
		if binary_flag == True:
			out_channels = 1
		else:
			out_channels = 2
		
		model = UNet(
			spatial_dims=3,
			in_channels=1,
			out_channels= out_channels,
			channels=(16, 32, 64, 128, 256),
			strides=(2, 2, 2, 2),
			num_res_units=2,
		)
	else:
		if binary_flag == True:
			out_channels = 1
		else:
			out_channels = 2
		model = UNETR(
			in_channels=1,
			out_channels=out_channels,
			img_size = img_size,
			feature_size = feature_size,
			hidden_size = hidden_size, 
			mlp_dim = mlp_dim, 
			num_heads = num_heads,
			pos_embed = pos_embed,
			norm_name = norm_name,
			res_block = res_block,
			dropout_rate = dropout_rate
		)

	return model