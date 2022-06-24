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
from scipy.io import savemat

# Add topology functions
# sys.path.insert(0,'...') # Modify topology path
# from clDice.cldice_metric.cldice import clDice as clDice_metric
from veela import clDice as clDice_metric

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

	return tqdm(range(1,args.epochs + 1), desc = 'Epoch X | X (Training loss: X) (Validation loss: X) (Validation '+ metric + ': X)', dynamic_ncols=True), loss, metric




def validation(model, val_loader, metric, criterions, post_trans, args):

	criterion_vessel = criterions[0]

	if len(args.decoder) == 2:
		criterion_dmap = criterions[1]
		criterion_ori = criterions[2]
	else:
		if args.decoder[0] == 'dmap':
			criterion_dmap = criterions[1]
		else:
			criterion_ori = criterions[1]


	dice_vals, haus_vals, avg_vals, cldice_vals = list(), list(), list(), list()
	cld_metric = list()
	loss_vals, loss_vessels = list(), list()

	val_loss = [loss_vals, loss_vessels]
	if len(args.decoder) == 2:
		loss_dmaps = list()
		loss_oris = list()
		val_loss.append(loss_dmaps)
		val_loss.append(loss_oris)
	else:
		if args.decoder[0] == 'dmap':
			loss_dmaps = list()
			val_loss.append(loss_dmaps)
		else:
			loss_oris = list()
			val_loss.append(loss_oris)

	model.eval()
	with torch.no_grad():
		for batch in val_loader:
			# val_outputs_ = sliding_window_inference(val_inputs, args.input_size, 4, model)
			if len(args.decoder) == 2:
				val_inputs, val_vessel_masks, val_dmap_masks, val_ori_masks = (batch['image'].to(device), batch['vessel'].to(device), batch['dmap'].to(device), batch['ori'].to(device))

				val_output_vessels_, val_output_dmap_, val_output_ori_ = model(val_inputs)

				loss_1 = criterion_vessel(val_output_vessels_, val_vessel_masks)
				loss_vessels.append(loss_1.item())
				loss_2 = criterion_dmap(val_output_dmap_, val_dmap_masks, val_vessel_masks)
				loss_dmaps.append(loss_2.item())
				loss_3 = criterion_ori(val_output_ori_, val_ori_masks)
				loss_oris.append(loss_3.item())

				# Plotear con twin axes para ver ratio
				loss = loss_1 + args.alpha_dmap * loss_2 + args.alpha_ori * loss_3
				loss_vals.append(loss.item())
			else:
				if args.decoder[0] == 'dmap':
					val_inputs, val_vessel_masks, val_dmap_masks = (batch['image'].to(device), batch['vessel'].to(device), batch['dmap'].to(device))

					val_output_vessels_, val_output_dmap_ = model(val_inputs)

					loss_1 = criterion_vessel(val_output_vessels_, val_vessel_masks)
					loss_vessels.append(loss_1.item())
					loss_2 = criterion_dmap(val_output_dmap_, val_dmap_masks, val_vessel_masks)
					loss_dmaps.append(loss_2.item())

					loss = loss_1 + args.alpha_dmap * loss_2
					loss_vals.append(loss.item())
				else:
					val_inputs, val_vessel_masks, val_ori_masks = (batch['image'].to(device), batch['vessel'].to(device), batch['ori'].to(device))
					val_output_vessels_, val_output_ori_ = model(val_inputs)
					
					loss_1 = criterion_vessel(val_output_vessels_, val_vessel_masks)
					loss_vessels.append(loss_1.item())
					loss_3 = criterion_ori(val_output_ori_, val_ori_masks)
					loss_oris.append(loss_3.item())

					# Plotear con twin axes para ver ratio
					loss = loss_1 + args.alpha_ori * loss_3
					loss_vals.append(loss.item())
			

			if args.binary:
				val_output_vessels = [post_trans(i) for i in decollate_batch(val_output_vessels_)]

				metric[0](y_pred=val_output_vessels, y=val_vessel_masks)
				dice_val = metric[0].aggregate().item()
				metric[1](y_pred=val_output_vessels, y=val_vessel_masks)
				haus_val = metric[1].aggregate().item()
				metric[2](y_pred=val_output_vessels, y=val_vessel_masks)
				avg_val = metric[2].aggregate().item()
				for output, label in zip(val_output_vessels, val_vessel_masks):
					clD = clDice_metric(output.squeeze().cpu().numpy().astype(bool), label.squeeze().cpu().numpy().astype(bool))
					cld_metric.append(clD)
				metric_val = np.mean(cld_metric)
				

			else:
				
				val_output_convert, val_labels_convert = utils.decollate_batch_list(val_output_vessels_, val_vessel_masks)

				metric[0](y_pred=val_output_convert, y=val_labels_convert)
				dice_val = metric[0].aggregate().item()
				metric[1](y_pred=val_output_convert, y=val_labels_convert)
				haus_val = metric[1].aggregate().item()
				metric[2](y_pred=val_output_convert, y=val_labels_convert)
				avg_val = metric[2].aggregate().item()
				for output, label in zip(val_output_convert, val_labels_convert):
					clD = clDice_metric(output.squeeze().argmax(dim =0).cpu().numpy().astype(bool), label.squeeze().argmax(dim =0).cpu().numpy().astype(bool))
					cld_metric.append(clD)
				metric_val = np.mean(cld_metric)	


				
			dice_vals.append(dice_val)
			haus_vals.append(haus_val)
			avg_vals.append(avg_val)
			cldice_vals.append(metric_val)


			[x.reset() for x in metric]
			# metric[0].reset()
			# metric[1].reset()
			# metric[2].reset()
			cld_metric = list()

	mean_dice_vals = np.mean(dice_vals)
	mean_haus_vals = np.mean(haus_vals)
	mean_avg_vals = np.mean(avg_vals)
	mean_cldice_vals = np.mean(cldice_vals)
	val_loss = list()
	mean_loss_val = np.mean(loss_vals)
	val_loss.append(mean_loss_val) # Total loss
	mean_loss_1 = np.mean(loss_vessels)
	val_loss.append(mean_loss_1) # Vessel loss
	if len(args.decoder) == 2:
		mean_loss_2 = np.mean(loss_dmaps)
		mean_loss_3 = np.mean(loss_oris)
		val_loss.append(mean_loss_2)
		val_loss.append(mean_loss_3)
	else:
		if args.decoder[0] == 'dmap':
			mean_loss_2 = np.mean(loss_dmaps)
			val_loss.append(mean_loss_2)
		else:
			mean_loss_3 = np.mean(loss_oris)
			val_loss.append(mean_loss_3)

	# print('\n\tValidation dice: {}\tValidation loss: {}'.format(mean_dice_val, mean_loss_val))

	return [mean_dice_vals, mean_haus_vals, mean_avg_vals, mean_cldice_vals], val_loss


def train(model, train_loader, val_loader, optimizer, metric, criterions, lossses_list_tr, lossses_list_val, metric_list, fold, args):
	dice_val_best = -1
	best_epoch = -1
	 
	# SELECT CRITERIONS
	criterion_vessel = criterions[0]
	if len(args.decoder) == 2:
		criterion_dmap = criterions[1]
		criterion_ori = criterions[2]
	else:
		if args.decoder[0] == 'dmap':
			criterion_dmap = criterions[1]
		else:
			criterion_ori = criterions[1]

	# criterion_vessel = criterions[0]
	# criterion_dmap = criterions[1]

	# epoch_iterator = tqdm(range(1,args.epochs + 1), desc = 'Epoch X | X (Training loss: X) (Validation loss: X) (Validation metric: X)', dynamic_ncols=True)
	epoch_iterator, loss_type, metric_type = get_epoch_iterator(args)

	for epoch in epoch_iterator:
		model.train()
		epoch_loss = 0
		vessel_loss, dmap_loss, ori_loss = 0,0,0
		step = 0

		for batch in train_loader:
			step += 1

			if len(args.decoder) == 2:
				inputs, vessel_masks, dmap_masks, ori_masks = (batch['image'].to(device), batch['vessel'].to(device), batch['dmap'].to(device), batch['ori'].to(device))
				optimizer.zero_grad()
				output_vessels, output_dmap, output_ori = model(inputs)

				loss_1 = criterion_vessel(output_vessels, vessel_masks)
				loss_2 = criterion_dmap(output_dmap, dmap_masks, vessel_masks)
				loss_3 = criterion_ori(output_ori, ori_masks)

				# NOT TESTED YET
				loss = loss_1 + loss_2 + loss_3

				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
				vessel_loss += loss_1.item()
				dmap_loss += loss_2.item()
				ori_loss += loss_3.item()
			else:
				if args.decoder[0] == 'dmap':
					inputs, vessel_masks, dmap_masks = (batch['image'].to(device), batch['vessel'].to(device), batch['dmap'].to(device))

					optimizer.zero_grad()
					output_vessels, output_dmap = model(inputs)

					loss_1 = criterion_vessel(output_vessels, vessel_masks)
					loss_2 = criterion_dmap(output_dmap, dmap_masks, vessel_masks)

					loss = loss_1 + args.alpha_dmap * loss_2

					loss.backward()
					optimizer.step()
					epoch_loss += loss.item()
					vessel_loss += loss_1.item()
					dmap_loss += loss_2.item()
				else:
					inputs, vessel_masks, ori_masks = (batch['image'].to(device), batch['vessel'].to(device), batch['ori'].to(device))
					optimizer.zero_grad()
					output_vessels, output_ori = model(inputs)

					loss_1 = criterion_vessel(output_vessels, vessel_masks)
					loss_3 = criterion_ori(output_ori, ori_masks)

					# Plotear con twin axes para ver ratio
					loss = loss_1 + args.alpha_ori * loss_3

					loss.backward()
					optimizer.step()
					epoch_loss += loss.item()
					vessel_loss += loss_1.item()
					ori_loss += loss_3.item()

		epoch_loss /= step
		vessel_loss /= step
		if len(args.decoder) == 2:
			dmap_loss /= step
			ori_loss /= step
		else:
			if args.decoder[0] == 'dmap':
				dmap_loss /= step
			else:
				ori_loss /= step

		lossses_list_tr[0].append(epoch_loss)
		lossses_list_tr[1].append(vessel_loss)
		if len(args.decoder) == 2:
			lossses_list_tr[2].append(dmap_loss)
			lossses_list_tr[3].append(ori_loss)
		else:
			if args.decoder[0] == 'dmap':
				lossses_list_tr[2].append(dmap_loss)
			else:
				lossses_list_tr[2].append(ori_loss)

		if (epoch  % eval_num == 0):
			metrics_val , losses_val = validation(model, val_loader, metric, criterions, post_trans, args)
			# loss_list.append(epoch_loss)
			metric_list[0].append(metrics_val[0])
			metric_list[1].append(metrics_val[1])
			metric_list[2].append(metrics_val[2])
			metric_list[3].append(metrics_val[3])
			lossses_list_val[0].append(losses_val[0])
			lossses_list_val[1].append(losses_val[1])
			lossses_list_val[2].append(losses_val[2]) # Either dmap loss or ori loss
			if len(args.decoder) == 2:
				lossses_list_val[3].append(losses_val[3])
			


			epoch_iterator.set_description('Epoch %d | %d (Training loss: %4f) (Validation loss: %4f) (Validation metric: %4f)' % (epoch,
																														   args.epochs,
																														   epoch_loss, 
																														   losses_val[0],
																														   metrics_val[0]))
			if metrics_val[0] > dice_val_best:
				best_epoch = epoch
				dice_val_best = metrics_val[0]

				utils.create_dir(os.path.join(os.path.abspath(os.getcwd()), 'weights', 'fold_'+str(fold)), remove_folder=True)
				torch.save(
					# model.state_dict(), os.path.join(os.path.join(os.path.abspath(os.getcwd()), 'weights'), "best_metric_model.pth")
					model.state_dict(), os.path.join(current_path,'weights', 'fold_'+str(fold),"best_metric_model.pth")
				)
		
	return lossses_list_tr, metric_list, lossses_list_val, best_epoch


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
	utils.create_dir(reshaped_liver_dir, remove_folder = False) # Create such directory where to pre-process dataset
	utils.split_dataset(info_dict, reshaped_liver_dir, args) # Preprocess-dataset, ready for training. Do nothing if already pre-processed. 

	json_routes, dictionary_list = utils.create_json_file(reshaped_liver_dir, info_dict, args)
	"""if args.binary:
		json_routes = [os.path.join('/home2/alberto/aux_TOPNET/code_TopNet/clDice','binary',('dmap' if (len(args.decoder) == 1 and args.decoder[0] == 'dmap') else 'ori' if (len(args.decoder) == 1 and args.decoder[0] == 'ori') else '3dec'),args.vessel,'VEELA_'+str(i)+'.json') for i in range(args.k)] # KFCV given splits
	else:
		json_routes = [os.path.join('/home2/alberto/aux_TOPNET/code_TopNet/clDice','multi','VEELA_'+str(i)+'.json') for i in range(args.k)] # KFCV given splits
	dictionary_list = [utils.json2dict(json_routes[i]) for i in range(args.k)] # KFCV given splits"""

	my_metrics = [DiceMetric(include_background=True if args.binary else False, reduction="mean", get_not_nans=False), 
				  HausdorffDistanceMetric(include_background=True if args.binary else False, reduction="mean", get_not_nans=False), 
				  SurfaceDistanceMetric(include_background=True if args.binary else False, reduction="mean", get_not_nans=False)]
	
	loss_list_tr, loss_list_ts, dice_list, haus_list, surface_list, cldice_list = list(), list(), list(), list(), list(), list()
	metric_list = list()
	metric_list.append(dice_list)
	metric_list.append(haus_list)
	metric_list.append(surface_list)
	metric_list.append(cldice_list)
	
	mean_folds, std_folds = list(), list()
	# loss_vessel_tr, loss_dmap_tr,loss_vessel_val, loss_dmap_val = list(), list(), list(), list()
	loss_vessel_tr = list()
	list_training = [[],loss_vessel_tr]

	loss_vessel_val = list()
	list_validation = [[], loss_vessel_val]

	if len(args.decoder) == 2:
		loss_dmap_tr = list()
		loss_ori_tr = list()

		loss_dmap_val = list()
		loss_ori_val = list()
		
		list_training.append(loss_dmap_tr)
		list_training.append(loss_ori_tr)

		list_validation.append(loss_dmap_val)
		list_validation.append(loss_ori_val)
	else:
		if args.decoder[0] == 'dmap':
			loss_dmap_tr = list()
			loss_dmap_val = list()

			list_training.append(loss_dmap_tr)
			list_validation.append(loss_dmap_val)
		else:
			loss_ori_tr = list()
			loss_ori_val = list()

			list_training.append(loss_ori_tr)
			list_validation.append(loss_ori_val)

	if not args.binary:
		mean_portal_folds, std_portal_folds, mean_hepatic_folds, std_hepatic_folds =  list(), list(), list(), list()

	utils.create_dir(os.path.join(os.path.abspath(os.getcwd()), 'weights'), remove_folder=False)
	metrics_all = np.zeros((args.k, 11 if len(args.decoder) == 2 else 10, args.epochs)) # Variable to save all the KFCV metrics
	
	
	for fold, (json_route, dictionary_) in enumerate( zip(json_routes, dictionary_list) ):
		print('Creating loaders fold {}...\n'.format(fold+1))
		train_loader, val_loader, test_loader = dataset_loader.get_loaders(args, json_route)

		# TYPICAL PYTORCH TRAINING PROCESS
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		print('Creating model...\n')
		model, criterions = models.get_model_loss(args)
		model.to(device)

		torch.backends.cudnn.benchmark = True
		optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		
		losses_tr, metric_list, losses_val, best_epoch = train(
			model,
			train_loader,
			val_loader,
			optimizer,
			my_metrics,
			criterions,
			list_training,
			list_validation,
			metric_list,
			fold,
			args)
		
		# SAVE METRICS

		metrics_all[fold,0,:] = np.asarray(losses_tr[0][-args.epochs:]) # Training loss (DiceCELoss + lambda * C_loss)
		metrics_all[fold,1,:] = np.asarray(losses_tr[1][-args.epochs:]) # Training DiceCELoss
		metrics_all[fold,2,:] = np.asarray(losses_tr[2][-args.epochs:]) # Training (C_loss (D2) or L1Loss (D3))
		if len(args.decoder) == 2:
			metrics_all[fold,3,:] = np.asarray(losses_tr[3][-args.epochs:]) # Training L1Loss

		metrics_all[fold,3,:] = np.asarray(losses_val[0][-args.epochs:]) # Validation loss (DiceCELoss + lambda * C_loss)
		metrics_all[fold,4,:] = np.asarray(losses_val[1][-args.epochs:]) # Validation DiceCELoss
		metrics_all[fold,5,:] = np.asarray(losses_val[2][-args.epochs:]) # Validation (C_loss (D2) or L1Loss (D3))
		if len(args.decoder) == 2:
			metrics_all[fold,6,:] = np.asarray(losses_val[2][-args.epochs:]) # Validation L1Loss
			metrics_all[fold,7,:] = np.asarray(metric_list[0][-args.epochs:]) # Dice Metric
			metrics_all[fold,8,:] = np.asarray(metric_list[1][-args.epochs:]) # Hausdorff distance metric
			metrics_all[fold,9,:] = np.asarray(metric_list[2][-args.epochs:]) # Average Surface Distance
			metrics_all[fold,10,:] = np.asarray(metric_list[3][-args.epochs:]) # Topology Metric (Cldice)
		else:

			metrics_all[fold,6,:] = np.asarray(metric_list[0][-args.epochs:]) # Dice Metric
			metrics_all[fold,7,:] = np.asarray(metric_list[1][-args.epochs:]) # Hausdorff distance metric
			metrics_all[fold,8,:] = np.asarray(metric_list[2][-args.epochs:]) # Average Surface Distance
			metrics_all[fold,9,:] = np.asarray(metric_list[3][-args.epochs:]) # Topology Metric (Cldice)

		# SAVE METRIC / LOSS PLOTS
		plots.save_loss_metric(losses_tr[0][-args.epochs:], metric_list[0][-args.epochs:], losses_val[0][-args.epochs:], fold, best_epoch, args) # CASO NORMAL
		# plots.save_loss_metric(losses_tr, metric_list[0][-args.epochs:], losses_val, fold, best_epoch, args) # Comparar pérdidas

		# INFERENCE ON TEST SET AND RESULT SAVING
		metrics = utils.pipeline_2(model,os.path.join(current_path, 'weights', 'fold_'+str(fold)), dictionary_, info_dict, test_loader, fold, args)
		mean_folds.append(metrics[0])
		std_folds.append(metrics[1])
		if not args.binary:
			mean_portal_folds.append(metrics[2])
			std_portal_folds.append(metrics[3])
			mean_hepatic_folds.append(metrics[4])
			std_hepatic_folds.append(metrics[5])
	
	# CREATE TXT FILE WITH METRICS
	with open('./metrics.txt', 'w') as f:
		if args.binary:
			f.write('Successfully performed K-fold Cross Validation with K = {}.\nBest prediction obtained in fold = {}\nKFCV-Mean dice: {}\nKFCV-Std dice: {}\n'.format(
				args.k, 
				np.argmax(mean_folds) + 1,
				np.mean(mean_folds),
				np.mean(std_folds)
			))
		else:
			f.write('Successfully performed K-fold Cross Validation with K = {}.\nBest prediction obtained in fold = {}\nKFCV-Portal Mean: {}\nKFCV-Portal Std: {}\nKFCV-Hepatic Mean: {}\nKFCV-Hepatic Std: {}\nKFCV-Mean: {}\nKFCV-Std: {}\n'.format(
				args.k, 
				np.argmax(mean_folds) + 1,
				np.mean(mean_portal_folds),
				np.mean(std_portal_folds),
				np.mean(mean_hepatic_folds),
				np.mean(std_hepatic_folds),
				np.mean(mean_folds),
				np.mean(std_folds)
			))

	save_matrix = {'metrics_all': metrics_all, 'label': 'Variable that contains train_loss, val_loss and val_metric for all splits'}
	savemat("metrics_all.mat", save_matrix)


if __name__ == '__main__':

	# PARSER WITH ALL ARGUMENTS
	parser = argparse.ArgumentParser(description='VEELA dataset segmentation with transformers')

	parser.add_argument('-dataset', required=False, type=str, default = 'VEELA') # Future: add choices
	parser.add_argument('-binary', required=False, type=str, default='True', choices=('True','False'),help = 'binary (True) or multi-class (False)')
	parser.add_argument('-vessel', required=False, type=str, default='portal', choices=('portal','hepatic'))
	parser.add_argument('-dataset_path', required = False, type=str, default='...', help = 'Define dataset path accordingly')
	parser.add_argument('-input_size', required=False, nargs='+', type = int, default=[224,224,128],help='Size of volume that feeds the network. Ex: --input_size 16 16 16')
	parser.add_argument('-batch', required=False, type=int, help='Batch size', default=1)
	parser.add_argument('-epochs', required=False, type=int, help='Number of epochs', default=2)
	parser.add_argument('-lr', required=False, type = float, help='Define learning rate', default=1e-4)
	parser.add_argument('-weight_decay', required=False, type=float, default=1e-5)
	parser.add_argument('-k', required=False, type=int, help='Number of folds for K-fold Cross Validation', default = 5)

	# parser.add_argument('-net', required=False, type=str, default='unetr', choices=('unet', 'unetr'))
	parser.add_argument('-decoder', required=False, nargs = '+', type=str, default='ori', choices=('dmap', 'ori', ['dmap', 'ori']), help = '(D1+D2), (D1+D3), (D1+D2+D3) architectures, respectively')
	parser.add_argument('-alpha_dmap', required=False, type=float, default=10, help='Weighting factor for C_loss.')
	parser.add_argument('-alpha_ori', required=False, type=float, default=50, help='Weighting factor for MSE.')
	parser.add_argument('-metric', required=False, type=str, default='dice', choices=('dice', 'haus', 'surfdist', 'softdice'))

	# UNETR ARGUMENTS (DO NOT TOUCH DEFAULT VALUES)
	parser.add_argument('-feature_size', required=False, type=int, default=16)
	parser.add_argument('-hidden_size', required=False, type=int, default = 768)
	parser.add_argument('-mlp_dim', required=False, type=int, default = 3072)
	parser.add_argument('-num_heads', required=False, type=int, default = 12)
	parser.add_argument('-pos_embed', required=False, type=str, default = 'perceptron', choices=('perceptron','conv'))
	parser.add_argument('-norm_name', required=False, type=str, default = 'instance', choices=('batch','instance'))
	parser.add_argument('-res_block', required=False, type=str, choices=('True','False'), default='True')
	parser.add_argument('-dropout_rate', required=False, type=float, default = 0)
	
	args = parser.parse_args()

	args.binary =  True if args.binary == 'True' else False
	args.res_block = True if args.res_block == 'True' else False

	args.input_size = tuple(args.input_size)
	main(args)
