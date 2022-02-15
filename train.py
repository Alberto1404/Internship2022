import torch
import argparse
import os
import numpy as np

import dataset_loader
import utils
import veela
import plots

from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.data import (
	DataLoader,
	CacheDataset,
	load_decathlon_datalist,
	decollate_batch,
)
from monai.transforms import AsDiscrete



def validation(model, global_step, epoch_iterator_val, dice_metric, post_label, post_pred, size):
	model.eval()
	dice_vals = list()
	with torch.no_grad():
		for step, batch in enumerate(epoch_iterator_val):
			val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
			val_outputs = sliding_window_inference(val_inputs, size, 4, model)
			val_labels_list = decollate_batch(val_labels)
			val_labels_convert = [
				post_label(val_label_tensor) for val_label_tensor in val_labels_list
			]
			val_outputs_list = decollate_batch(val_outputs)
			val_output_convert = [
				post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
			]
			dice_metric(y_pred=val_output_convert, y=val_labels_convert)
			dice = dice_metric.aggregate().item()
			dice_vals.append(dice)
			epoch_iterator_val.set_description(
				"Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
			)
		dice_metric.reset()
	mean_dice_val = np.mean(dice_vals)
	return mean_dice_val


def train(model, size,global_step, train_loader, val_loader,dice_val_best, global_step_best, optimizer, dice_metric, loss_function, eval_num, post_label, post_pred, max_iterations):
	model.train()
	epoch_loss = 0
	step = 0
	epoch_loss_values = list()
	metric_values = list()

	epoch_iterator = tqdm(
		train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
	)
	for step, batch in enumerate(epoch_iterator):
		step += 1
		x, y = (batch["image"].cuda(), batch["label"].cuda())
		logit_map = model(x)
		loss = loss_function(logit_map, y)
		loss.backward()
		epoch_loss += loss.item()
		optimizer.step()
		optimizer.zero_grad()
		epoch_iterator.set_description(
			"Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
		)
		if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:

			epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
			dice_val = validation(model, global_step, epoch_iterator_val, dice_metric, post_label, post_pred, size)
			epoch_loss /= step
			epoch_loss_values.append(epoch_loss)
			metric_values.append(dice_val)
			if dice_val > dice_val_best:
				dice_val_best = dice_val
				global_step_best = global_step
				utils.create_dir(os.path.join(os.path.abspath(os.getcwd()), 'weights'))
				torch.save(
					# model.state_dict(), os.path.join(os.path.join(os.path.abspath(os.getcwd()), 'weights'), "best_metric_model.pth")
					model.load_state_dict(torch.load(os.path.join(os.path.abspath(os.getcwd()), 'weights', "best_metric_model.pth")))
				)
				print(
					"Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
						dice_val_best, dice_val
					)
				)
			else:
				print(
					"Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
						dice_val_best, dice_val
					)
				)
		global_step += 1
	return global_step, dice_val_best, global_step_best, epoch_loss_values, metric_values


def main(args):

	# SETUP TRANSFORMS FOR TRAINING AND VALIDATION
	# Transformations called in dataset loader. 

	print('Processing dataset...\n')
	# DOWNLOAD DATASET AND FORMAT IN THE FOLDER
	info_dict = veela.process_dataset(args.dataset_path)
	# dst_folder = './data_reshaped' # Folder to save resized images
	dst_folder = os.path.join(os.path.abspath(os.getcwd()), 'data_reshaped')
	utils.create_dir(dst_folder)
	print('Splitting dataset...\n')
	veela.split_dataset(info_dict, args.dataset_path, args.input_size, dst_folder)

	# LOAD DATASET
	print('Creating JSON file...\n')
	json_routes, dictionary_list = utils.create_json_file(dst_folder, info_dict, args.k)
	print('Creating train and valid loaders...\n')
	train_loader, val_loader, val_ds = dataset_loader.get_train_valid_loader(args.input_size, json_routes)


	# CREATE MODEL, LOSS, OPTIMIZER
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Creating model...\n')
	if args.binary == True:
		model = UNETR(
			in_channels=1,
			out_channels=2,
			img_size = args.input_size,
			feature_size=args.feature_size,
			hidden_size=args.hidden_size, # 1920  768
			mlp_dim = args.mlp_dim, # 7680   3072
			num_heads = args.num_heads,
			pos_embed = args.pos_embed,
			norm_name = args.norm_name,
			res_block = args.res_block,
			dropout_rate = args.dropout_rate
		).to(device)
	else:
		model = UNETR(
			in_channels=1,
			out_channels=3,
			img_size = args.input_size,
			feature_size=args.feature_size,
			hidden_size=args.hidden_size, # 1920  768
			mlp_dim = args.mlp_dim, # 7680   3072
			num_heads = args.num_heads,
			pos_embed = args.pos_embed,
			norm_name = args.norm_name,
			res_block = args.res_block,
			dropout_rate = args.dropout_rate
		).to(device)

	loss_function = DiceCELoss(softmax=True, to_onehot_y=True)
	torch.backends.cudnn.benchmark = True
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

	# EXECUTE A TYPICAL PYTORCH TRAINING PROCESS
	# max_iterations = 50000
	max_iterations = args.max_it
	eval_num = 500
	post_label = AsDiscrete(to_onehot = 2)
	post_pred = AsDiscrete(argmax=True, to_onehot = 2)
	dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
	global_step = 0
	dice_val_best = 0.0
	global_step_best = 0
	while global_step < max_iterations:
		global_step, dice_val_best, global_step_best, epoch_loss_values, metric_values = train(
			model, 
			args.input_size,
			global_step, 
			train_loader, 
			val_loader,
			dice_val_best, 
			global_step_best,
			optimizer,
			dice_metric,
			loss_function,
			eval_num, 
			post_label,
			post_pred,
			args.max_it)
	# model.load_state_dict(torch.load(os.path.join(os.path.join(os.path.abspath(os.getcwd()), 'weights'), "best_metric_model.pth")))
	model.load_state_dict(torch.load(os.path.join(os.path.abspath(os.getcwd()), 'weights', "best_metric_model.pth")))

	print(f"Train completed, best_metric: {dice_val_best:.4f} "f"at iteration: {global_step_best}")

	# CHECK BEST MODEL OUTPUT WITH THE INPUT IMAGE AND LABEL
	plots.plot_slices(model,val_ds, args.input_size, os.path.join(os.path.abspath(os.getcwd()), 'weights'), dictionary_list[0]) # BAD!!!!!!!! K-FOLD CROSS VALIDATION !!!!!
	plots.plot_loss_metric(eval_num,epoch_loss_values, metric_values)

	# SAVE SEGMENTATIONS
	utils.save_segmentations(model,os.path.join(os.path.abspath(os.getcwd()), 'weights'), dictionary_list[0], info_dict, args.input_size, val_ds)



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='VEELA dataset segmentation with transformers')

	parser.add_argument('-binary', required=False, type=str, default='yes', choices=('yes','no'))
	parser.add_argument('-dataset_path', required = False, type=str, default='/home/guijosa/Documents/PythonDocs/UNETR/VEELA/dataset')
	parser.add_argument('-batch', required=False, type=int, help='Batch size', default=1)
	parser.add_argument('-max_it', required=False, type=int, help='Number of iterations', default=25000)
	parser.add_argument('-lr', required=False, type = float, help='Define learning rate', default=1e-4)
	parser.add_argument('-weight_decay', required=False, type=float, default=1e-5)
	parser.add_argument('-k', required=False, type=int, help='Number of folds for K-fold Cross Validation', default = 1)
	# UNETR
	parser.add_argument('--input_size', required=False, nargs='+', type = int, default=[128,128,128],help='Size of volume that feeds the network. Ex: --input_size 16 16 16')
	parser.add_argument('-feature_size', required=False, type=int, default=16)
	parser.add_argument('-hidden_size', required=False, type=int, default = 768)
	parser.add_argument('-mlp_dim', required=False, type=int, default = 3072)
	parser.add_argument('-num_heads', required=False, type=int, default = 16)
	parser.add_argument('-pos_embed', required=False, type=str, default = 'perceptron',choices=('perceptron','conv'))
	parser.add_argument('-norm_name', required=False, type=str, default = 'instance',choices=('batch','instance'))
	parser.add_argument('-res_block', required=False, type=str, choices=('yes','no'), default='yes')
	parser.add_argument('-dropout_rate', required=False, type=float, default = 0.0)
	
	args = parser.parse_args()

	if args.binary == 'yes':
		args.binary = True
	else:
		args.binary = False

	if args.res_block == 'yes':
		args.res_block = True
	else:
		args.res_block = False
	
	# Cast input size to tuple 
	args.input_size = tuple(args.input_size)
	main(args)