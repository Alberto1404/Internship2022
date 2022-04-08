from secrets import choice
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
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric

# Add topology functions
sys.path.insert(0,os.path.join(os.path.abspath(os.getcwd()), 'clDice')) # Modify topology path
from clDice.cldice_metric.cldice import clDice as clDice_metric

from monai.inferers import sliding_window_inference
from monai.data import (
	decollate_batch,
)
from monai.transforms import AsDiscrete,  Activations, EnsureType, Compose

# Global variables
post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])
post_pred = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=3)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_path = os.path.abspath(os.getcwd())
eval_num = 1

def get_epoch_iterator(args):

	if args.metric == 'softdice':
		loss = 'Soft dice ClDice loss'
		metric = 'clDice metric'
	else:
		if args.metric == 'dice':
			metric = args.metric
		elif args.metric == 'haus':
			metric = 'Hausdorff distance'
		elif args.metric == 'surfdist':
			metric = 'Average surface distance'

		loss = 'DiceCELoss'

	return tqdm(range(1,args.epochs + 1), desc = 'Epoch X | X (Training '+ loss +': X) (Validation '+ loss + ': X) (Validation '+ metric + ': X)', dynamic_ncols=True), loss, metric

def validation(model, val_loader, metric, criterion_vessel, criterion_dmap, post_trans, args):

	metric_vals = list()
	loss_vals = list()
	loss_vessels, loss_dmaps = list(), list()
	cld_metric = list()

	model.eval()
	with torch.no_grad():
		for batch in val_loader:
			val_inputs, val_vessel_masks, val_dmap_masks = (batch['image'].to(device), batch['vessel'].to(device), batch['dmap'].to(device))
			# val_outputs_ = sliding_window_inference(val_inputs, args.input_size, 4, model)
			val_output_vessels_, val_output_dmap_ = model(val_inputs)

			if args.binary:
				val_output_vessels = [post_trans(i) for i in decollate_batch(val_output_vessels_)]
				
				if args.metric == 'softdice':
					for output, label in zip(val_outputs, val_labels):
						clD = clDice_metric(output.squeeze().cpu().numpy().astype(bool), label.squeeze().cpu().numpy().astype(bool))
						cld_metric.append(clD)
					metric_val = np.mean(cld_metric)
				else:
					metric(y_pred=val_output_vessels, y=val_vessel_masks)
					metric_val = metric.aggregate().item()

			else:
				"""val_labels_list = decollate_batch(val_labels)
				val_labels_convert = [
					post_label(val_label_tensor) for val_label_tensor in val_labels_list
				]
				val_outputs_list = decollate_batch(val_outputs_)
				val_output_convert = [
					post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
				]"""
				val_output_convert, val_labels_convert = utils.decollate_batch_list(val_outputs_, val_labels)

				if args.metric == 'softdice':
					for output, label in zip(val_output_convert, val_labels_convert):
						clD = clDice_metric(output.squeeze().argmax(dim =0).cpu().numpy().astype(bool), label.squeeze().argmax(dim =0).cpu().numpy().astype(bool))
						# clD = clDice_metric(val_output_convert[0].squeeze().argmax(dim =0).cpu().numpy().astype(bool), val_labels_convert[0].squeeze().argmax(dim =0).cpu().numpy().astype(bool))
						cld_metric.append(clD)
					metric_val = np.mean(cld_metric)				
				else:
					metric(y_pred=val_output_convert, y=val_labels_convert)
					metric_val = metric.aggregate().item()
				
			metric_vals.append(metric_val)

			if args.metric == 'softdice':
				loss = loss_function(val_labels, val_outputs_)
				# loss_vals.append(loss)
			else:
				loss_1 = criterion_vessel(val_output_vessels_, val_vessel_masks)
				loss_vessels.append(loss_1.item())
				loss_2 = criterion_dmap(val_output_dmap_, val_dmap_masks, val_vessel_masks)
				loss_dmaps.append(loss_2.item())
				loss = args.alpha*loss_1 + (1-args.alpha) * loss_2

			loss_vals.append(loss.item())

		if args.metric != 'softdice':
			metric.reset()
		else:
			cld_metric = list()
	mean_metric_vals = np.mean(metric_vals)
	mean_loss_val = np.mean(loss_vals)
	mean_loss_1 = np.mean(loss_vessels)
	mean_loss_2 = np.mean(loss_dmaps)
	# print('\n\tValidation dice: {}\tValidation loss: {}'.format(mean_dice_val, mean_loss_val))

	return mean_metric_vals, mean_loss_val, mean_loss_1, mean_loss_2


def train(model, train_loader, val_loader, optimizer, metric, losses, loss_list_tr, loss_list_tr_vessel, loss_list_tr_dmap, loss_vessel_val, loss_dmap_val, loss_list_ts, metric_list, args):
	dice_val_best = 0.0
	criterion_vessel = losses[0]
	criterion_dmap = losses[1]

	# epoch_iterator = tqdm(range(1,args.epochs + 1), desc = 'Epoch X | X (Training loss: X) (Validation loss: X) (Validation metric: X)', dynamic_ncols=True)
	epoch_iterator, loss_type, metric_type = get_epoch_iterator(args)

	for epoch in epoch_iterator:
		model.train()
		epoch_loss = 0
		vessel_loss, dmap_loss = 0,0
		step = 0

		for batch in train_loader:
			step += 1
			inputs, vessel_masks, dmap_masks = (batch['image'].to(device), batch['vessel'].to(device), batch['dmap'].to(device))
			optimizer.zero_grad()
			output_vessels, output_dmap = model(inputs)
			# loss = loss_function(outputs, labels)
			if args.metric == 'softdice': 
				# ClDice requires the Ground-Truth first
				# loss = loss_function(labels, outputs)
				print("TIENES QUE MODIFICAR ESTO")
			else:
				loss_1 = criterion_vessel(output_vessels, vessel_masks)
				loss_2 = criterion_dmap(output_dmap, dmap_masks, vessel_masks)

				# loss = args.alpha*loss_1 + (1-args.alpha) * loss_2
				loss = loss_1 + args.alpha * loss_2

			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
			vessel_loss += loss_1.item()
			dmap_loss += loss_2.item()

		epoch_loss /= step
		vessel_loss /= step
		dmap_loss /= step

		loss_list_tr.append(epoch_loss)
		loss_list_tr_vessel.append(vessel_loss)
		loss_list_tr_dmap.append(dmap_loss)

		if (epoch  % eval_num == 0):
			dice_val , loss_val, loss_val_1,loss_val_2 = validation(model, val_loader, metric, criterion_vessel, criterion_dmap, post_trans, args)
			# loss_list.append(epoch_loss)
			metric_list.append(dice_val)
			loss_list_ts.append(loss_val)
			loss_vessel_val.append(loss_val_1)
			loss_dmap_val.append(loss_val_2)


			epoch_iterator.set_description('Epoch %d | %d (Training %s: %4f) (Validation %s: %4f) (Validation %s: %4f)' % (epoch,
																														   args.epochs,
																														   loss_type,
																														   epoch_loss, 
																														   loss_type,
																														   loss_val,
																														   metric_type,
																														   dice_val))
			if dice_val > dice_val_best:
				# best_before = dice_val_best
				# dice_val_best = dice_val
				utils.create_dir(os.path.join(os.path.abspath(os.getcwd()), 'weights'), remove_folder=True)
				torch.save(
					# model.state_dict(), os.path.join(os.path.join(os.path.abspath(os.getcwd()), 'weights'), "best_metric_model.pth")
					model.state_dict(), os.path.join(current_path,'weights',"best_metric_model.pth")
				)
				"""print(
					"Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
						dice_val_best, best_before
					)
				)
			else:
				print(
					"Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
						dice_val_best, best_before
					)
				)"""
		
	return loss_list_tr, metric_list, loss_list_ts, [loss_list_tr_vessel, loss_list_tr_dmap], [loss_vessel_val, loss_dmap_val]


def main(args):
	
	try: # Find DATASET_NAME.json in data directory, and directly load it if previously read from previous trainings
		glob.glob(current_path + '/' + args.dataset + '.json')[0]
		print('Found ' + args.dataset + '.json at ' + current_path +'. Loading info...')
		dict_names = utils.json2dict(current_path + '/' + args.dataset + '.json')
	except IndexError:
		print('No ' + args.dataset + '.JSON found. Proceed reading ' + args.dataset + ' from ' + args.dataset_path + '...')
		dict_names = veela.get_names_from_dataset(args.dataset_path, args.dataset, current_path)
		utils.dict2json(dict_names, args.dataset, current_path)
	
	# LOAD DATASET
	info_dict = veela.load_dataset(dict_names, args.dataset_path)
	reshaped_liver_dir = os.path.join(current_path, 'data_reshaped_' + 
												  str(args.input_size[0]) +
												  'x' + str(args.input_size[1]) +
												  'x' + str(args.input_size[2]) + 
												  ('_binary' if args.binary else '_multi')
	) # Folder to save resized images that feed the network								
	
	print('Splitting dataset... \n')
	utils.create_dir(reshaped_liver_dir, remove_folder = False)
	utils.split_dataset(info_dict, reshaped_liver_dir, args)

	# json_routes, dictionary_list = utils.create_json_file(reshaped_liver_dir, info_dict, args)
	json_routes = os.path.join(reshaped_liver_dir,'VEELA_0.json')
	dictionary_list = utils.json2dict(json_routes)

	print('Creating loaders...\n')
	train_loader, val_loader, test_loader = dataset_loader.get_loaders(args, json_routes)


	# CREATE MODEL, LOSS, OPTIMIZER
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	print('Creating model...\n')
	model, loss_function_vessel, loss_function_dmap = models.get_model_loss(args)
	model.to(device)

	torch.backends.cudnn.benchmark = True
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	# EXECUTE A TYPICAL PYTORCH TRAINING PROCESS
	"""if not args.cldice:
		dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
	else:
		dice_metric = None"""
	if args.metric == 'haus':
		metric = HausdorffDistanceMetric(include_background=True if args.binary else False)
	elif args.metric == 'surfdist':
		metric = SurfaceDistanceMetric(include_background=True if args.binary else False)
	elif args.metric == 'dice':
		metric = DiceMetric(include_background=True if args.binary else False, reduction="mean", get_not_nans=False)
	else: # Cldice case, cldice metric is not instantiable object to further call
		metric = None
		# DiceMetric(include_background=True if args.binary else False, reduction="mean", get_not_nans=False)
	loss_list_tr, loss_list_ts, metric_list = list(), list(), list()
	loss_vessel_tr, loss_dmap_tr,loss_vessel_val, loss_dmap_val = list(), list(), list(), list()

	loss_list_tr, metric_list, loss_list_ts, losses_1_2_train, losses_1_2_val = train(
		model,
		train_loader,
		val_loader,
		optimizer,
		metric,
		[loss_function_vessel, loss_function_dmap],
		loss_list_tr,
		loss_vessel_tr, 
		loss_dmap_tr,
		loss_vessel_val,
		loss_dmap_val,
		loss_list_ts,
		metric_list,
		args
	)
	
	# SAVE ACCURACY / LOSS PLOTS
	# plots.save_loss_metric(loss_list_tr, metric_list, loss_list_ts, args) # ANTES ESTABA ASI
	plots.save_loss_metric(loss_list_tr, metric_list, loss_list_ts, losses_1_2_train, losses_1_2_val, args)

	# SAVE SEGMENTATIONS
	# utils.pipeline_2(model,os.path.join(current_path, 'weights'), dictionary_list[0], info_dict, args.dataset_path, test_loader, args.batch)
	utils.pipeline_2(model,os.path.join(current_path, 'weights'), dictionary_list, info_dict, test_loader, args)

if __name__ == '__main__':

	
	parser = argparse.ArgumentParser(description='VEELA dataset segmentation with transformers')

	parser.add_argument('-dataset', required=False, type=str, default = 'VEELA') # Future: add choices
	parser.add_argument('-binary', required=False, type=str, default='True', choices=('True','False'))
	parser.add_argument('-dataset_path', required = False, type=str, default='/home/guijosa/Documents/PythonDocs/VEELA/dataset')
	parser.add_argument('-input_size', required=False, nargs='+', type = int, default=[224,224,128],help='Size of volume that feeds the network. Ex: --input_size 16 16 16')
	parser.add_argument('-batch', required=False, type=int, help='Batch size', default=1)
	parser.add_argument('-epochs', required=False, type=int, help='Number of epochs', default=500)
	parser.add_argument('-lr', required=False, type = float, help='Define learning rate', default=1e-4)
	parser.add_argument('-weight_decay', required=False, type=float, default=1e-5)
	parser.add_argument('-k', required=False, type=int, help='Number of folds for K-fold Cross Validation', default = 5)

	parser.add_argument('-net', required=False, type=str, default='unetr', choices=('unet', 'unetr'))
	parser.add_argument('-alpha', required=False, type=float, default=0.5, help='Weighting factor for loss functions. Value between 0 and 1. Alpha * DiceCELoss + (1-alpha) * C_loss')
	# parser.add_argument('-cldice', required=False, type=str, default='False', choices=('False', 'True'))
	parser.add_argument('-metric', required=False, type=str, default='dice', choices=('dice', 'haus', 'surfdist', 'softdice'))

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

	# if selected metric is softdice, cldice must be set to True
	args.binary =  True if args.binary == 'True' else False
	args.res_block = True if args.res_block == 'True' else False
	# args.cldice = True if args.cldice == 'True' else False

	args.input_size = tuple(args.input_size)
	main(args)
