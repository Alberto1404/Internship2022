import os 
import json
import numpy as np
import random
from scipy.fft import dst
import torch
import skimage.transform as skTrans
import nibabel as nib
import glob
import statistics

import veela

from medpy.metric.binary import dc as DiceMetric
from tqdm import tqdm
from pathlib import Path
from monai.config import PathLike
from typing import Dict, List
from monai.data import decollate_batch
from monai.transforms import AsDiscrete,  Activations, EnsureType, Compose

post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_dataset(info_dict, dst_folder, args):
	if not os.path.exists(dst_folder) or len(os.listdir(dst_folder)) == 0:
		veela.pipeline_1(info_dict, dst_folder,args)
	else:
		print('Dataset already splitted.\n')


def create_dir(path, remove_folder):
	# If directory does not exists
	if not os.path.exists(path):
		os.makedirs(path)

	# If it does: 
	# - remove_folder = False -> Do nothing
	# - remove_folder = True -> Remove the content of the folder as well as the folder, and recursively calls itself to create the folder again
	if remove_folder:
		for file in os.listdir(path):
			os.remove(os.path.join(path,file))
		os.rmdir(path)
		create_dir(path, False)

def get_index_dict(my_dict):

	all_list = list()
	for element in my_dict['Image name']:
		all_list.append(element.split('-')[0])

	return all_list

def kfcv(dataset, k):
	print('Performing K-Fold Cross-Validation... ')
	data_backup = dataset.copy() # Dataset is the list with all the idxs

	folds_training = np.zeros((23,k))
	folds_validation = np.zeros((5,k))
	# folds_test = np.zeros((7,k))
	folds_test = np.reshape(sorted(random.sample(dataset,7)) * k, (5,7)).T # Expected shape: (7,5)

	for fold in range(k):
		validation = sorted(random.sample(dataset,5))
		while any(item in folds_test[:,fold] for item in validation):
			validation = sorted(random.sample(dataset,5))
		training = sorted(list(set(dataset) - set(folds_test[:,fold]) - set(validation)))

		folds_training[:,fold] = training
		folds_validation[:,fold] = validation

	"""print('Performing K-Fold Cross-Vailidation... ')
	data_backup = dataset.copy() # Dataset is the list with all the idxs

	folds_training = np.zeros((23,k))
	folds_validation = np.zeros((5,k))
	folds_test = np.zeros((7,k))
	
	for fold in range(k):
		training = sorted(random.sample(dataset,23))
		print('\nTraining indexed obtained, obtaining validation... ')
		validation = sorted(random.sample(dataset,5))
		while any(item in training for item in validation):
			validation = sorted(random.sample(dataset,5))
		print('\nValidation indexes obtained!!!')
		test = sorted(list(set(dataset) - set(training) - set(validation)))

		folds_training[:,fold] = training
		folds_validation[:,fold] = validation
		folds_test[:,fold] = test"""

	return folds_training.astype(int), folds_validation.astype(int), folds_test.astype(int)

def json2dict(directory):
	with open(directory) as json_file:
		dictionary = json.load(json_file)
	return dictionary

def dict2json(info_dict, dataset_name, save_dir):

	# save_dir = 'home2/alberto/data' # '...'
	# Serializing json 
	json_object = json.dumps(info_dict, indent = 4)

	# Writing to sample.json
	with open(os.path.join(save_dir, dataset_name) + '.json', 'w') as outfile:
		outfile.write(json_object)


def create_json_file(dst_folder, info_dict, args): # Add segmentation flag for future

	indexes_list = get_index_dict(info_dict)
	
	folds_training, folds_validation, folds_test = kfcv(indexes_list, args.k)
	json_routes = list()
	dictionary_list = list()
	ending = '-VE-liver_multi_GT' if args.binary == False else '-VE-liver_por_GT' # Define as desired

	for fold in tqdm(range(np.shape(folds_training)[1])):
		# Data to be written
		dictionary = {
			"name": args.dataset,
			"test": [
				{
					"image": os.path.join(dst_folder,str(folds_test[0, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_test[0, fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[1, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_test[1, fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[2, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_test[2, fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[3, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_test[3, fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[4, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_test[4, fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[5, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_test[5, fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[6, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_test[6, fold]).zfill(3) + ending + '.nii.gz')
				}
			],
			"training": [
				{
					"image": os.path.join(dst_folder,str(folds_training[0, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[0,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[1, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[1,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[2, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[2,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[3, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[3,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[4, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[4,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[5, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[5,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[6, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[6,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[7, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[7,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[8, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[8,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[9, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[9,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[10, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[10,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[11, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[11,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[12, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[12,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[13, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[13,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[14, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[14,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[15, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[15,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[16, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[16,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[17, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[17,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[18, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[18,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[19, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[19,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[20, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[20,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[21, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[21,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[22, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_training[22,fold]).zfill(3) + ending + '.nii.gz')
				}
			],
			"validation": [
				{
					"image": os.path.join(dst_folder,str(folds_validation[0, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_validation[0,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_validation[1, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_validation[1,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_validation[2, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_validation[2,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_validation[3, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_validation[3,fold]).zfill(3) + ending + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_validation[4, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"label": os.path.join(dst_folder,str(folds_validation[4,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
				}
			]
		}
		dictionary.update({"labels": {"0": "background", "1": "portal_vessels", "2": "hepatic_vessels"}}) if args.binary == False else dictionary.update({"labels": {"0": "background", "1": "portal_vessels"}})
		# Serializing json 
		json_object = json.dumps(dictionary, indent = 4)

		# Writing to sample.json
		json_routes.append(os.path.join(dst_folder,dictionary['name']) +'_'+str(fold)+ '.json')
		dictionary_list.append(dictionary)
		with open(os.path.join(dst_folder,dictionary['name']) +'_'+str(fold)+ '.json', 'w') as outfile:
			outfile.write(json_object)

	return json_routes, dictionary_list


def get_position(my_dict, key_dict, myvalue):
	
	for idx, element in enumerate(my_dict[key_dict]):
		if myvalue.split('-')[0] in element:
			return idx


def get_list_of_pos(json_dict, info_dict, key):
	idxlist = list()
	for idx, data in enumerate(json_dict['test']):
		idxlist.append(get_position(info_dict, key, data['image'].split('/')[-1]))

	return idxlist

def pipeline_2(model,weights_dir, json_dict, info_dict, test_loader, args):
	output_route = os.path.join(os.path.abspath(os.getcwd()), 'results')
	create_dir(output_route, remove_folder=True)
	dices_portal = list()
	dices_hepatic = list()
	idxlist = get_list_of_pos(json_dict, info_dict, 'Image name')
	model.load_state_dict(torch.load(os.path.join(weights_dir, "best_metric_model.pth")))
	model.eval()

	with torch.no_grad():
		for test_data in test_loader:
			test_images, val_labels = test_data["image"].to(device), test_data["label"].to(device)
			pred = model(test_images)
			pred = post_trans(decollate_batch(pred)) # Length of pred accordingly to batchsize
			
			for result in pred:

				# NETWORK OUTPUT
				result = result.squeeze().cpu().numpy()
				pos = idxlist[0]
				if not args.binary:

					# NETWORK OUTPUT 
					portal = result[1,:,:,:]
					hepatic = result[2,:,:,:]

					# UNRESIZE TO OWN LIVER SIZE + BINARIZATION CAUSED BY RESIZING
					unresized_portal = veela.resize(portal, info_dict, pos)
					unresized_hepatic = veela.resize(hepatic, info_dict, pos)

					unresized_portal = veela.binarize(unresized_portal,1)
					unresized_hepatic = veela.binarize(unresized_hepatic,2)

					# INTRODUCE SEGMENTED LIVER IN BLACK VOLUME
					result_portal = veela.index_liver_in_volume(unresized_portal, info_dict, pos)
					result_hepatic = veela.index_liver_in_volume(unresized_hepatic, info_dict, pos)

					# NUMPY 2 NIFTI
					output_portal = nib.Nifti1Image(result_portal, info_dict['Affine matrix'][pos], info_dict['Header'][pos])
					output_hepatic = nib.Nifti1Image(result_hepatic, info_dict['Affine matrix'][pos], info_dict['Header'][pos])
					output_multi = nib.Nifti1Image(result_portal + result_hepatic, info_dict['Affine matrix'][pos], info_dict['Header'][pos])

					portal_gt = nib.load(os.path.join(args.dataset_path, info_dict['Portal veins name'][pos])).get_fdata()
					hepatic_gt = nib.load(os.path.join(args.dataset_path, info_dict['Hepatic veins name'][pos])).get_fdata()
					gt_multi = nib.Nifti1Image(veela.binarize(portal_gt,1) + veela.binarize(hepatic_gt,2), info_dict['Affine matrix'][pos], info_dict['Header'][pos])

					# DICE METRIC COMPUTATION ON TEST LOADER
					dice_portal = 100.*DiceMetric(result_portal.astype(np.bool), portal_gt.astype(np.bool))
					dice_hepatic = 100.*DiceMetric(result_hepatic.astype(np.bool), hepatic_gt.astype(np.bool))
					dices_portal.append(dice_portal)
					dices_hepatic.append(dice_hepatic)
					print('Dice metric for portal veins: {} %\nDice metric for hepatic veins: {} %\n'.format(dice_portal, dice_hepatic))

					nib.save(output_portal, output_route + '/' + info_dict['Image name'][pos] + '_por_segmented.nii.gz')
					nib.save(output_hepatic, output_route + '/' + info_dict['Image name'][pos] + '_hep_segmented.nii.gz')
					nib.save(output_multi, output_route + '/' + info_dict['Image name'][pos] + '_join_segmented.nii.gz')
					nib.save(gt_multi, output_route + '/' + info_dict['Image name'][pos] + '_gt_join_segmented.nii.gz')

				else:
					# UNRESIZE TO OWN LIVER SIZE + BINARIZATION CAUSED BY RESIZING
					unresized_result = veela.resize(result, info_dict, pos)
					unresized_result = veela.binarize(unresized_result)

					# INTRODUCE SEGMENTED LIVER IN BLACK VOLUME
					result = veela.index_liver_in_volume(unresized_result, info_dict, pos)

					# NUMPY 2 NIFTI
					output_ima = nib.Nifti1Image(result, info_dict['Affine matrix'][pos], info_dict['Header'][pos])

					groundtruth = nib.load(os.path.join(args.dataset_path, info_dict['Portal veins name'][pos])).get_fdata()
					# Compute dice metric
					dice = 100.*DiceMetric(result.astype(np.bool), groundtruth.astype(np.bool))
					dices_portal.append(dice)
					print('Dice metric: {} %'.format(dice))

					nib.save(output_ima, output_route + '/' + info_dict['Image name'][pos] + '_segmented.nii.gz')
				idxlist.pop(0)
	

	with open('./dices.txt', 'w') as f:
		if args.binary:
			f.write('Mean: {}\nStd: {}\n'.format(np.mean(dices_portal), statistics.stdev(dices_portal)))
		else:
			f.write('Portal mean: {}\nStd: {}\nHepatic mean: {}\nStd: {}\nTotal mean: {}\nStd: {}\n'.format(
				np.mean(dices_portal), statistics.stdev(dices_portal),
				np.mean(dices_hepatic), statistics.stdev(dices_hepatic),
				np.mean(dices_portal+dices_hepatic), statistics.stdev(dices_portal + dices_hepatic)
			))


def load_veela_datalist(data_list_file_path: PathLike, data_list_key: str = "training") -> List[Dict]:

	data_list_file_path = Path(data_list_file_path)
	if not data_list_file_path.is_file():
		raise ValueError(f"Data list file {data_list_file_path} does not exist.")
	with open(data_list_file_path) as json_file:
		json_data = json.load(json_file)
	if data_list_key not in json_data:
		raise ValueError(f'Data list {data_list_key} not specified in "{data_list_file_path}".')
	expected_data = json_data[data_list_key]
	return expected_data