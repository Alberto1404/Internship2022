# Library imports
import torch
import argparse
import os
import numpy as np
import glob

# Dependent file imports
import dataset_loader
import utils
import veela
import plots
import models

# Module imports
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import (
	decollate_batch,
)
from monai.transforms import AsDiscrete,  Activations, EnsureType, Compose

# Global variables
post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_path = os.path.abspath(os.getcwd())
eval_num = 500

# Functions
def validation(model, global_step, epoch_iterator_val, dice_metric, post_label, post_pred, size):
	model.eval()
	dice_vals = list()
	with torch.no_grad():
		for step, batch in enumerate(epoch_iterator_val):
			val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
			val_outputs = sliding_window_inference(val_inputs, size, 4, model)
			val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
			# val_labels_list = decollate_batch(val_labels)
			# val_labels_convert = [
			# 	post_label(val_label_tensor) for val_label_tensor in val_labels_list
			# ]
			# val_outputs_list = decollate_batch(val_outputs)
			# val_output_convert = [
			# 	post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
			# ]
			dice_metric(y_pred=val_outputs, y=val_labels)
			dice = dice_metric.aggregate().item()
			dice_vals.append(dice)
			epoch_iterator_val.set_description(
				"Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
			)
		dice_metric.reset()
	mean_dice_val = np.mean(dice_vals)
	return mean_dice_val


def train(model, size, train_loader, val_loader, optimizer, dice_metric, loss_function, post_label, post_pred, global_step, max_iterations):
	model.train()
	dice_val_best = 0.0
	global_step_best = 0
	epoch_loss = 0
	step = 0
	epoch_loss_values = list()
	metric_values = list()

	epoch_iterator = tqdm(
		train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
	)
	for step, batch in enumerate(epoch_iterator):
		step += 1
		x, y = (batch["image"].to(device), batch["label"].to(device))
		optimizer.zero_grad()
		logit_map = model(x)
		loss = loss_function(logit_map, y)
		loss.backward()
		epoch_loss += loss.item()
		optimizer.step()
		
		epoch_iterator.set_description(
			"Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
		)
		if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations: # FOR SAVING

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
					model.state_dict(), os.path.join(current_path,'weights',"best_metric_model.pth")
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

	try: # Find DATASET_NAME.json in data directory, and directly load it if previously read from previous trainings
		print('Finding ' + args.dataset + ' dictionary..')
		glob.glob(current_path + '/' + args.dataset + '.json')[0]
		print('Found ' + args.dataset + '.json at ' + current_path +'. Loading...')
		dict_names = utils.open_json_file(current_path + '/' + args.dataset + '.json')
	except IndexError:
		print('No ' + args.dataset + '.JSON found. Proceed reading ' + args.dataset + ' from ' + args.dataset_path + '...')
		dict_names = veela.get_names_from_dataset(args.dataset_path, args.dataset, current_path)
		utils.save_dataset_dict(dict_names, args.dataset, current_path)
	
	info_dict = veela.process_dataset(dict_names, args.dataset_path)
	dst_folder = os.path.join(current_path, 'data_reshaped') # Folder to save resized images that feed the network
	utils.create_dir(dst_folder)
	
	print('Splitting dataset...\n')
	veela.split_dataset(info_dict, args.dataset_path, args.input_size, dst_folder)

	# LOAD DATASET
	print('Creating JSON file...\n')
	json_routes, dictionary_list = utils.create_json_file(dst_folder, info_dict, args.k)
	print('Creating train and valid loaders...\n')
	train_loader, val_loader, test_loader, val_ds = dataset_loader.get_train_valid_loader(args.input_size, json_routes)


	# CREATE MODEL, LOSS, OPTIMIZER
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	print('Creating model...\n')
	model = models.get_model(args.net, args.binary, args.input_size, args.feature_size, args.hidden_size, args.mlp_dim, args.num_heads, args.pos_embed, args.norm_name, args.res_block, args.dropout_rate)
	
	loss_function = DiceCELoss(sigmoid=True, to_onehot_y=False)
	torch.backends.cudnn.benchmark = True
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

	# EXECUTE A TYPICAL PYTORCH TRAINING PROCESS
	# max_iterations = 50000
	max_iterations = args.max_it
	post_label = AsDiscrete(to_onehot = 2)
	post_pred = AsDiscrete(argmax=True, to_onehot = 2)
	dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
	global_step = 0

	while global_step < max_iterations:
		global_step, dice_val_best, global_step_best, epoch_loss_values, metric_values = train(
			model, 
			args.input_size,
			train_loader, 
			val_loader,
			optimizer,
			dice_metric,
			loss_function,
			post_label,
			post_pred,
			global_step,
			args.max_it)

	print(f"Train completed, best_metric: {dice_val_best:.4f} "f"at iteration: {global_step_best}")

	# CHECK BEST MODEL OUTPUT WITH THE INPUT IMAGE AND LABEL
	# plots.plot_slices(model,val_ds, args.input_size, os.path.join(current_path, 'weights'), dictionary_list[0]) # BAD!!!!!!!! K-FOLD CROSS VALIDATION !!!!!
	plots.save_loss_metric(eval_num,epoch_loss_values, metric_values)

	# SAVE SEGMENTATIONS
	utils.save_segmentations(model,os.path.join(current_path, 'weights'), dictionary_list[0], info_dict, args.input_size, test_loader)



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='VEELA dataset segmentation with transformers')

	parser.add_argument('-dataset', required=False, type=str, default = 'VEELA') # Future: add choices
	parser.add_argument('-binary', required=False, type=str, default='True', choices=('True','False'))
	parser.add_argument('-dataset_path', required = False, type=str, default='home2/alberto/data/VEELA/dataset')
	parser.add_argument('-batch', required=False, type=int, help='Batch size', default=1)
	parser.add_argument('-max_it', required=False, type=int, help='Number of iterations', default=1000)
	parser.add_argument('-lr', required=False, type = float, help='Define learning rate', default=1e-4)
	parser.add_argument('-weight_decay', required=False, type=float, default=1e-5)
	parser.add_argument('-k', required=False, type=int, help='Number of folds for K-fold Cross Validation', default = 1)

	parser.add_argument('-net', required=False, type=str, default='unetr', choices=('unet', 'unetr'))

	# UNETR
	parser.add_argument('--input_size', required=False, nargs='+', type = int, default=[128,128,128],help='Size of volume that feeds the network. Ex: --input_size 16 16 16')
	parser.add_argument('-feature_size', required=False, type=int, default=16)
	parser.add_argument('-hidden_size', required=False, type=int, default = 768)
	parser.add_argument('-mlp_dim', required=False, type=int, default = 3072)
	parser.add_argument('-num_heads', required=False, type=int, default = 16)
	parser.add_argument('-pos_embed', required=False, type=str, default = 'perceptron', choices=('perceptron','conv'))
	parser.add_argument('-norm_name', required=False, type=str, default = 'instance', choices=('batch','instance'))
	parser.add_argument('-res_block', required=False, type=str, choices=('True','False'), default='True')
	parser.add_argument('-dropout_rate', required=False, type=float, default = 0.0)
	
	args = parser.parse_args()

	if args.binary == 'True':
		args.binary = True
	else:
		args.binary = False

	if args.res_block == 'True':
		args.res_block = True
	else:
		args.res_block = False
	
	# Cast input size to tuple 
	args.input_size = tuple(args.input_size)
	main(args)
