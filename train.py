import torch
import argparse
import os
import numpy as np
import glob
import sys

# Dependent file imports
import dataset_loader
import utils
import veela
import plots
import models


# Module imports
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import (
	decollate_batch,
)
from monai.transforms import AsDiscrete,  Activations, EnsureType, Compose

# Global variables
post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=3)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_path = os.path.abspath(os.getcwd())
eval_num = 1

def validation(model, val_loader, dice_metric, loss_function, post_label, post_pred, post_trans, args):

	dice_vals = list()
	loss_vals = list()
	model.eval()
	with torch.no_grad():
		for batch in val_loader:
			val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
			val_outputs_ = sliding_window_inference(val_inputs, args.input_size, 4, model)
			if args.binary:
				val_outputs = [post_trans(i) for i in decollate_batch(val_outputs_)]
				dice_metric(y_pred=val_outputs, y=val_labels)

			else:
				val_labels_list = decollate_batch(val_labels)
				val_labels_convert = [
				  post_label(val_label_tensor) for val_label_tensor in val_labels_list
				]
				val_outputs_list = decollate_batch(val_outputs_)
				val_output_convert = [
				  post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
				]
				dice_metric(y_pred=val_output_convert, y=val_labels_convert)

			
			
			
			dice = dice_metric.aggregate().item()
			dice_vals.append(dice)
			# loss = loss_function(val_outputs_, val_labels)
			if args.cldice == True:
				loss = loss_function(val_labels, val_outputs)
			else:
				loss = loss_function(val_outputs_, val_labels)
			loss_vals.append(loss.item())

			
		dice_metric.reset()
	mean_dice_val = np.mean(dice_vals)
	mean_loss_val = np.mean(loss_vals)
	print('\n\tValidation dice: {}\tValidation loss: {}'.format(mean_dice_val, mean_loss_val))

	return mean_dice_val, mean_loss_val


def train(model, train_loader, val_loader, optimizer, dice_metric, loss_function, loss_list_tr, loss_list_ts, metric_list, args):
	dice_val_best = 0.0
	best_before = 0.0

	for epoch in tqdm(range(1,args.epochs + 1), desc = 'Training...'):
		print("-" * 110)
		print('Epoch {}/{} '.format(epoch, args.epochs))
		model.train()
		epoch_loss = 0
		step = 0
		

		for batch in train_loader:
			step += 1
			inputs, labels = (batch['image'].to(device), batch['label'].to(device))
			optimizer.zero_grad()
			outputs = model(inputs)
			# loss = loss_function(outputs, labels)
			if args.cldice == True: 
				# ClDice requires the Ground-Truth first
				loss = loss_function(labels, outputs)
			else:
				loss = loss_function(outputs, labels)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()

		epoch_loss /= step
		loss_list_tr.append(epoch_loss)
		print('\taverage loss: {}'.format(epoch_loss))

		if (epoch  % eval_num == 0):
			dice_val , loss_val= validation(model, val_loader, dice_metric, loss_function, post_label, post_pred, post_trans, args)
			# loss_list.append(epoch_loss)
			metric_list.append(dice_val)
			loss_list_ts.append(loss_val)

			if dice_val > dice_val_best:
				best_before = dice_val_best
				dice_val_best = dice_val
				utils.create_dir(os.path.join(os.path.abspath(os.getcwd()), 'weights'), remove_folder=True)
				torch.save(
					# model.state_dict(), os.path.join(os.path.join(os.path.abspath(os.getcwd()), 'weights'), "best_metric_model.pth")
					model.state_dict(), os.path.join(current_path,'weights',"best_metric_model.pth")
				)
				print(
					"Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
						dice_val_best, best_before
					)
				)
			else:
				print(
					"Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
						dice_val_best, best_before
					)
				)
		
	return loss_list_tr, metric_list, loss_list_ts


def main(args):
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
	dst_folder = os.path.join(current_path, 'data_reshaped_' + str(args.input_size[0]) + 'x' +str(args.input_size[1]) + 'x' +str(args.input_size[2]) + ('_binary' if args.binary == True else '_multi')) # Folder to save resized images that feed the network
	utils.create_dir(dst_folder, remove_folder = False) # COMMENT IF NEEEDED
	
	print('Splitting dataset... \n')
	utils.split_dataset(info_dict, dst_folder, args)
	"""veela.split_dataset(info_dict, args.input_size, dst_folder,args) # COMMENT IF NEEEDED"""
	

	print('Creating JSON file...\n')
	# json_routes, dictionary_list = utils.create_json_file(dst_folder, info_dict, args)

	json_routes = os.path.join(dst_folder,'VEELA_0.json')
	dictionary_list = utils.open_json_file(json_routes)

	###################################################################################################

	print('Creating train and valid loaders...\n')
	train_loader, val_loader, test_loader, val_ds = dataset_loader.get_train_valid_loader(args, json_routes)


	# CREATE MODEL, LOSS, OPTIMIZER
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	print('Creating model...\n')
	model, loss_function = models.get_model(args)
	model.to(device)

	torch.backends.cudnn.benchmark = True
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	# EXECUTE A TYPICAL PYTORCH TRAINING PROCESS
	dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
	loss_list_tr, loss_list_ts, metric_list = list(), list(), list()

	# loss_list_tr, metric_list, loss_list_ts = train(model,train_loader, val_loader, optimizer, dice_metric, loss_function, loss_list_tr, loss_list_ts, metric_list, args)
	# print(f"Train completed, best_metric: {dice_val_best:.4f} "f"at iteration: {global_step_best}")

	
	# SAVE ACCURACY / LOSS PLOTS
	# plots.save_loss_metric(loss_list_tr, metric_list, loss_list_ts)

	# SAVE SEGMENTATIONS
	# utils.save_segmentations(model,os.path.join(current_path, 'weights'), dictionary_list[0], info_dict, args.dataset_path, test_loader, args.batch)
	utils.save_segmentations(model,os.path.join(current_path, 'weights'), dictionary_list, info_dict, test_loader, args)

if __name__ == '__main__':

	
	parser = argparse.ArgumentParser(description='VEELA dataset segmentation with transformers')

	parser.add_argument('-dataset', required=False, type=str, default = 'VEELA') # Future: add choices
	parser.add_argument('-binary', required=False, type=str, default='False', choices=('True','False'))
	parser.add_argument('-dataset_path', required = False, type=str, default='/home/guijosa/Documents/PythonDocs/UNETR/VEELA/dataset')
	parser.add_argument('--input_size', required=False, nargs='+', type = int, default=[256,256,128],help='Size of volume that feeds the network. Ex: --input_size 16 16 16')
	parser.add_argument('-batch', required=False, type=int, help='Batch size', default=1)
	parser.add_argument('-epochs', required=False, type=int, help='Number of epochs', default=250)
	parser.add_argument('-lr', required=False, type = float, help='Define learning rate', default=1e-4)
	parser.add_argument('-weight_decay', required=False, type=float, default=1e-5)
	parser.add_argument('-k', required=False, type=int, help='Number of folds for K-fold Cross Validation', default = 5)

	parser.add_argument('-net', required=False, type=str, default='unet', choices=('unet', 'unetr'))
	parser.add_argument('-cldice', required=False, type=str, default='False', choices=('False', 'True'))

	# UNETR
	parser.add_argument('-feature_size', required=False, type=int, default=16)
	parser.add_argument('-hidden_size', required=False, type=int, default = 768)
	parser.add_argument('-mlp_dim', required=False, type=int, default = 3072)
	parser.add_argument('-num_heads', required=False, type=int, default = 12)
	parser.add_argument('-pos_embed', required=False, type=str, default = 'perceptron', choices=('perceptron','conv'))
	parser.add_argument('-norm_name', required=False, type=str, default = 'instance', choices=('batch','instance'))
	parser.add_argument('-res_block', required=False, type=str, choices=('True','False'), default='True')
	parser.add_argument('-dropout_rate', required=False, type=float, default = 0.0)
	
	args = parser.parse_args()

	args.binary =  True if args.binary == 'True' else False
	args.res_block = True if args.res_block == 'True' else False
	args.cldice = True if args.cldice == 'True' else False

	args.input_size = tuple(args.input_size)
	main(args)