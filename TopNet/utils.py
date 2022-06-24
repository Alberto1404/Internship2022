import os 
import json
import numpy as np
import random
import torch
import skimage.transform as skTrans
import nibabel as nib
import statistics
import sys

import veela

from medpy.metric.binary import dc as DiceMetric_bin, hd as HausdorffDistanceMetric_bin, asd as SurfaceDistanceMetric_bin
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from tqdm import tqdm
from pathlib import Path
from monai.config import PathLike
from monai.inferers import sliding_window_inference
from typing import Dict, List
from monai.data import decollate_batch
from monai.transforms import AsDiscrete,  Activations, EnsureType, Compose

# Add topology functions
# sys.path.insert(0,'...') # Modify topology path
# from clDice.cldice_metric.cldice import clDice as clDice_metric
from veela import clDice as clDice_metric


post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])
post_pred = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=3)])

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

def get_index_dict(my_dict): # Get index ID from CT & masks names

	all_list = list()
	for element in my_dict['Image name']:
		all_list.append(element.split('-')[0])

	return all_list

def kfcv(dataset, k):
	print('Performing K-Fold Cross-Validation... ')
	data_backup = dataset.copy() # Dataset is the list with all the idxs
	
	# Fixed size of training-validation-test for VEELA dataset!!
	folds_training = np.zeros((23,k))
	folds_validation = np.zeros((5,k))
	folds_test = np.zeros((7,k))
	
	# Ensure randomness in KFCV
	np.random.shuffle(dataset)

	for fold in range(k): # Rotation of 7 per fold -> K = 5. Modify as desired. 
		folds_training[:,fold] = np.roll(dataset,-k*fold)[:23]
		folds_validation[:,fold] = np.roll(dataset,-k*fold)[23:28]
		folds_test[:,fold] = np.roll(dataset,-k*fold)[-7:]
	return folds_training.astype(int), folds_validation.astype(int), folds_test.astype(int)

def json2dict(directory):
	with open(directory) as json_file:
		dictionary = json.load(json_file)
	return dictionary

def dict2json(info_dict, dataset_name, save_dir):

	# Serializing json 
	json_object = json.dumps(info_dict, indent = 4)

	# Writing to sample.json
	with open(os.path.join(save_dir, dataset_name) + '.json', 'w') as outfile:
		outfile.write(json_object)


def create_json_file(dst_folder, info_dict, args): # THIS JSON FILE WILL BE READ BY MONAI VIA "LOADCACHEDATASET" IN ORDER TO CREATE THE TRAIN-VAL-TEST LOADERS

	indexes_list = get_index_dict(info_dict)
	
	folds_training, folds_validation, folds_test = kfcv(indexes_list, args.k)
	json_routes = list()
	dictionary_list = list()
	ending_1 = '-VE-liver_multi_vessel' if args.binary == False else ('-VE-liver_por_vessel' if args.vessel == 'portal' else '-VE-liver_hep_vessel') # Define as desired
	ending_2 = '-VE-liver_multi_dmap' if args.binary == False else ('-VE-liver_por_dmap' if args.vessel == 'portal' else '-VE-liver_hep_dmap') # Define as desired
	ending_3 = '-VE-liver_por_ori' if args.vessel == 'portal' else '-VE-liver_hep_ori' # Define as desired

	for fold in tqdm(range(np.shape(folds_training)[1])):
		# Data to be written
		dictionary = {
			"name": args.dataset,
			"test": [
				{
					"image": os.path.join(dst_folder,str(folds_test[0, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_test[0, fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_test[0, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_test[0, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[1, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_test[1, fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_test[1, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_test[1, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[2, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_test[2, fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_test[2, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_test[2, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[3, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_test[3, fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_test[3, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_test[3, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[4, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_test[4, fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_test[4, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_test[4, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[5, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_test[5, fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_test[5, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_test[5, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_test[6, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_test[6, fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_test[6, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_test[6, fold]).zfill(3) + ending_3 + '.nii.gz')
				}
			],
			"training": [
				{
					"image": os.path.join(dst_folder,str(folds_training[0, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[0,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[0, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[0, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[1, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[1,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[1, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[1, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[2, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[2,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[2, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[2, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[3, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[3,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[3, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[3, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[4, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[4,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[4, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[4, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[5, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[5,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[5, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[5, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[6, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[6,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[6, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[6, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[7, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[7,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[7, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[7, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[8, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[8,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[8, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[8, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[9, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[9,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[9, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[9, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[10, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[10,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[10, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[10, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[11, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[11,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[11, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[11, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[12, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[12,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[12, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[12, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[13, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[13,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[13, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[13, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[14, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[14,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[14, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[14, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[15, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[15,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[15, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[15, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[16, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[16,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[16, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[16, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[17, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[17,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[17, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[17, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[18, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[18,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[18, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[18, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[19, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[19,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[19, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[19, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[20, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[20,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[20, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[20, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[21, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[21,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[21, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[21, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_training[22, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_training[22,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_training[22, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_training[22, fold]).zfill(3) + ending_3 + '.nii.gz')
				}
			],
			"validation": [
				{
					"image": os.path.join(dst_folder,str(folds_validation[0, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_validation[0,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_validation[0, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_validation[0, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_validation[1, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_validation[1,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_validation[1, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_validation[1, fold]).zfill(3) + ending_3 + '.nii.gz')
					
				},
				{
					"image": os.path.join(dst_folder,str(folds_validation[2, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_validation[2,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_validation[2, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_validation[2, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_validation[3, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_validation[3,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_validation[3, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_validation[3, fold]).zfill(3) + ending_3 + '.nii.gz')
				},
				{
					"image": os.path.join(dst_folder,str(folds_validation[4, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
					"vessel": os.path.join(dst_folder,str(folds_validation[4,fold]).zfill(3) + ending_1 + '.nii.gz'),
					"dmap": os.path.join(dst_folder,str(folds_validation[4, fold]).zfill(3) + ending_2 + '.nii.gz'),
					"ori": os.path.join(dst_folder,str(folds_validation[4, fold]).zfill(3) + ending_3 + '.nii.gz')
				}
			]
		}
		# dictionary.update({"labels": {"0": "background", "1": "portal_vessels", "2": "hepatic_vessels"}}) if args.binary == False else dictionary.update({"labels": {"0": "background", "1": "portal_vessels"}})
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

def decollate_batch_list(prob ,test_labels): # USED ONLY FOR MULTICLASS, NOT NECESSARY FOR D1+D2 / D1+D3

	test_labels_list = decollate_batch(test_labels)
	test_labels = [
		post_label(test_label_tensor) for test_label_tensor in test_labels_list
	] # List of B (batch_size) elements, each is a tensor of (3, H,W,D)
	test_outputs_list = decollate_batch(prob)
	prediction = [
		post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list
	] # List of B (batch_size) elements, each is a tensor of (3, H,W,D)

	return prediction, test_labels

def compute_metric_binary(y_pred, y, metric):

	if metric == 'haus':
		# metric_bin = HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)
		metric = HausdorffDistanceMetric_bin(y_pred.cpu().numpy().astype(bool), y.cpu().numpy().astype(bool))
	elif metric == 'surfdist':
		# metric_bin = SurfaceDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)
		metric = SurfaceDistanceMetric_bin(y_pred.cpu().numpy().astype(bool), y.cpu().numpy().astype(bool))
	elif metric == 'dice':
		# metric_bin = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
		metric = DiceMetric_bin(y_pred.cpu().numpy().astype(bool), y.cpu().numpy().astype(bool))

	return metric

def save_predictions(prediction, idxlist, output_route, info_dict, k, args):
	for input in prediction:
		pos = idxlist[0]

		if args.binary:
			input = input.squeeze().cpu().numpy().astype(np.float32)
		else:
			input = input.squeeze().cpu().numpy().argmax(axis = 0).astype(np.float32)

		# UNRESIZE TO OWN LIVER SIZE + BINARIZATION CASUED BY RESIZING
		unresized_result = veela.resize(input, info_dict, pos)
		
		if not args.binary: # "BINARIZATION" FOR MULTICLASS
			unresized_result = np.round(unresized_result)
		else:
			unresized_result = veela.binarize(unresized_result)
		

		# INTRODUCE SEGMENTED LIVER IN BLACK VOLUME
		result = veela.index_liver_in_volume(unresized_result, info_dict, pos)

		# NUMPY 2 NIFTI
		output = nib.Nifti1Image(result, info_dict['Affine matrix'][pos], info_dict['Header'][pos])
		if not args.binary:
			nib.save(output, output_route + '/fold_'+str(k) + '/' + info_dict['Image name'][pos] + '_join_segmented.nii.gz')
			
			# For better comparison
			portal = nib.load(os.path.join(args.dataset_path, info_dict['Portal veins name'][pos])).get_fdata()
			hepatic = nib.load(os.path.join(args.dataset_path, info_dict['Hepatic veins name'][pos])).get_fdata()
			gt = nib.Nifti1Image(veela.binarize(portal,1) + veela.binarize(hepatic,2), info_dict['Affine matrix'][pos], info_dict['Header'][pos])
			nib.save(gt, output_route + '/fold_'+str(k) + '/' + info_dict['Image name'][pos] + '_gt_join.nii.gz')
		else:
			nib.save(output, output_route + '/fold_'+str(k) + '/' + info_dict['Image name'][pos] + '_por_segmented.nii.gz')
		idxlist.pop(0)

def pipeline_2(model,weights_dir, json_dict, info_dict, test_loader, k, args): 
	# FROM OUTPUT OF THE NETWORK, TO FINAL NIFTI VOLUMES FOR VISUALIZATION
	
	output_route = os.path.join(os.path.abspath(os.getcwd()), 'results')
	create_dir(output_route, remove_folder=False)
	create_dir(output_route + '/fold_'+str(k), remove_folder=True)
	dices_test, dices_test_p, dices_test_h, hauss_test, hauss_test_p, hauss_test_h, avgs_test, avgs_test_p, avgs_test_h = list(),list(), list(), list(),list(), list(), list(),list(), list()
	cldice_vals, cldice_vals_p, cldice_vals_h, cld_metric, cld_metric_p, cld_metric_h = list(), list(), list(), list(), list(), list()

	metrics_multi = [DiceMetric(include_background=False, reduction="mean", get_not_nans=False), 
		HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False), 
		SurfaceDistanceMetric(include_background=False, reduction="mean", get_not_nans=False)]
	idxlist = get_list_of_pos(json_dict, info_dict, 'Image name')
	model.load_state_dict(torch.load(os.path.join(weights_dir, "best_metric_model.pth")))
	model.eval()

	with torch.no_grad():
		for test_data in test_loader:
			test_inputs, test_vessel_masks = (test_data['image'].to(device), test_data['vessel'].to(device))
			# prob = sliding_window_inference(test_inputs, args.input_size, 4, model) # torch.Size([2, 3, 224, 224, 128])
			test_output_vessels_, _ = model(test_inputs)

			if args.binary:
				prediction = [post_trans(i) for i in decollate_batch(test_output_vessels_)]

				dice = compute_metric_binary(torch.stack(prediction), test_vessel_masks, 'dice')
				haus = compute_metric_binary(torch.stack(prediction), test_vessel_masks, 'haus')
				avg = compute_metric_binary(torch.stack(prediction), test_vessel_masks, 'surfdist')
				for output, label in zip(prediction, test_vessel_masks):
					clD = clDice_metric(output.squeeze().cpu().numpy().astype(bool), label.squeeze().cpu().numpy().astype(bool))
					cld_metric.append(clD)
				cldice = np.mean(cld_metric)
					

			else:
				# NOT TESTED... 
				
				# Both portal and hepatic (include backgroun False)
				# prediction_b, test_labels_b = decollate_batch_list(prob[:,1:,:,:,:], test_labels)
				prediction, test_labels_ = decollate_batch_list(test_output_vessels_, test_vessel_masks)
				# Portal only
				prediction_p, test_labels_p = decollate_batch_list(test_output_vessels_[:,1,:,:,:].unsqueeze(dim=1), test_vessel_masks)
				# Hepatic only
				prediction_h, test_labels_h = decollate_batch_list(test_output_vessels_[:,2,:,:,:].unsqueeze(dim=1), test_vessel_masks)

				metrics_multi[0](y_pred=prediction, y=test_labels_)
				dice = metrics_multi[0].aggregate().item()
				metrics_multi[1](y_pred=prediction, y=test_labels_)
				haus = metrics_multi[1].aggregate().item()
				metrics_multi[2](y_pred=prediction, y=test_labels_)
				avg = metrics_multi[2].aggregate().item()
				for output, label in zip(prediction, test_labels_):
					clD = clDice_metric(output.squeeze().argmax(dim=0).cpu().numpy().astype(bool), label.squeeze().argmax(dim =0).cpu().numpy().astype(bool))
					cld_metric.append(clD)
				cldice = np.mean(cld_metric)


				dice_test_p = compute_metric_binary(torch.stack(prediction)[:,1,:,:,:], torch.stack(test_labels_)[:,1,:,:,:], 'dice')
				dice_test_h = compute_metric_binary(torch.stack(prediction)[:,2,:,:,:], torch.stack(test_labels_)[:,2,:,:,:], 'dice')
				haus_test_p = compute_metric_binary(torch.stack(prediction)[:,1,:,:,:], torch.stack(test_labels_)[:,1,:,:,:], 'haus')
				haus_test_h = compute_metric_binary(torch.stack(prediction)[:,2,:,:,:], torch.stack(test_labels_)[:,2,:,:,:], 'haus')
				avg_test_p = compute_metric_binary(torch.stack(prediction)[:,1,:,:,:], torch.stack(test_labels_)[:,1,:,:,:], 'surfdist')
				avg_test_h = compute_metric_binary(torch.stack(prediction)[:,2,:,:,:], torch.stack(test_labels_)[:,2,:,:,:], 'surfdist')

				for output, label in zip(torch.stack(prediction)[:,1,:,:,:], torch.stack(label)[:,1,:,:,:]):
					clD = clDice_metric(output.squeeze().argmax(dim=0).cpu().numpy().astype(bool), label.squeeze().argmax(dim =0).cpu().numpy().astype(bool))
					# clD = clDice_metric(val_output_convert[0].squeeze().argmax(dim =0).cpu().numpy().astype(bool), val_labels_convert[0].squeeze().argmax(dim =0).cpu().numpy().astype(bool))
					cld_metric_p.append(clD)
				cldice_p = np.mean(cld_metric_p)

				for output, label in zip(torch.stack(prediction)[:,2,:,:,:], torch.stack(label)[:,2,:,:,:]):
					clD = clDice_metric(output.squeeze().argmax(dim=0).cpu().numpy().astype(bool), label.squeeze().argmax(dim =0).cpu().numpy().astype(bool))
					# clD = clDice_metric(val_output_convert[0].squeeze().argmax(dim =0).cpu().numpy().astype(bool), val_labels_convert[0].squeeze().argmax(dim =0).cpu().numpy().astype(bool))
					cld_metric_h.append(clD)
				cldice_h = np.mean(cld_metric_h)


				dices_test_p.append(dice_test_p)
				dices_test_h.append(dice_test_h)
				hauss_test_p.append(haus_test_p)
				hauss_test_h.append(haus_test_h)
				avgs_test_p.append(avg_test_p)
				avgs_test_h.append(avg_test_h)
				cldice_vals_p.append(cldice_p)
				cldice_vals_h.append(cldice_h)
				

			dices_test.append(dice)
			hauss_test.append(haus)
			avgs_test.append(avg)
			cldice_vals.append(cldice)

			metrics_multi[0].reset()
			metrics_multi[1].reset()
			metrics_multi[2].reset()
			cld_metric, cld_metric_p, cld_metric_h = list(), list(), list()

			# SAVE SEGMENTATIONS
			save_predictions(prediction, idxlist, output_route, info_dict, k, args)
	# SAVE METRICS

	if args.binary:
		dice_mean = np.mean(dices_test) # total_mean = np.mean(metrics_l[0])
		dice_std = statistics.stdev(dices_test) # total_std = statistics.stdev(metrics_l[0])
		haus_mean = np.mean(hauss_test) # total_mean = np.mean(metrics_l[0])
		haus_std = statistics.stdev(hauss_test) # total_std = statistics.stdev(metrics_l[0])
		avg_mean = np.mean(avgs_test) # total_mean = np.mean(metrics_l[0])
		avg_std = statistics.stdev(avgs_test) # total_std = statistics.stdev(metrics_l[0]) 
		cldice_mean = np.mean(cldice_vals) 
		cldice_std = statistics.stdev(cldice_vals)

		with open('./metrics_'+str(k)+'.txt', 'w') as f:
			f.write('Dice Mean: {}\nDice Std: {}\nHausdorff Distance Mean: {}\nHausdorff Distance Std: {}\nAverage Surface Distance Mean: {}\nAverage Surface Distance Std: {}\nCldice mean: {}\nCldice std: {}\n'.format(dice_mean, dice_std, haus_mean, haus_std, avg_mean, avg_std, cldice_mean, cldice_std))

			return [dice_mean, dice_std, haus_mean, haus_std, avg_mean, avg_std, cldice_mean, cldice_std]
	else:
		dice_portal_mean = np.mean(dices_test_p) # portal_mean = np.mean(metrics_l[1])
		dice_portal_std = statistics.stdev(dices_test_p) # portal_std = statistics.stdev(metrics_l[1])
		dice_hepatic_mean = np.mean(dices_test_h) # hepatic_mean = np.mean(metrics_l[2])
		dice_hepatic_std = statistics.stdev(dices_test_h) # hepatic_std = statistics.stdev(metrics_l[2])
		haus_portal_mean = np.mean(hauss_test_p) # portal_mean = np.mean(metrics_l[1])
		haus_portal_std = statistics.stdev(hauss_test_p) # portal_std = statistics.stdev(metrics_l[1])
		haus_hepatic_mean = np.mean(hauss_test_h) # hepatic_mean = np.mean(metrics_l[2])
		haus_hepatic_std = statistics.stdev(hauss_test_h) # hepatic_std = statistics.stdev(metrics_l[2])
		avg_portal_mean = np.mean(avgs_test_p) # portal_mean = np.mean(metrics_l[1])
		avg_portal_std = statistics.stdev(avgs_test_p) # portal_std = statistics.stdev(metrics_l[1])
		avg_hepatic_mean = np.mean(avgs_test_h) # hepatic_mean = np.mean(metrics_l[2])
		avg_hepatic_std = statistics.stdev(avgs_test_h) # hepatic_std = statistics.stdev(metrics_l[2])
		cldice_portal_mean = np.mean(cldice_vals_p)
		cldice_portal_std = statistics.stdev(cldice_vals_p)
		cldice_hepatic_mean = np.mean(cldice_vals_h)
		cldice_hepatic_std = statistics.stdev(cldice_vals_h)

		dice_mean = np.mean(dices_test) # total_mean = np.mean(metrics_l[0])
		dice_std = statistics.stdev(dices_test) # total_std = statistics.stdev(metrics_l[0])
		haus_mean = np.mean(hauss_test) # total_mean = np.mean(metrics_l[0])
		haus_std = statistics.stdev(hauss_test) # total_std = statistics.stdev(metrics_l[0])
		avg_mean = np.mean(avgs_test) # total_mean = np.mean(metrics_l[0])
		avg_std = statistics.stdev(avgs_test) # total_std = statistics.stdev(metrics_l[0])
		cldice_mean = np.mean(cldice_vals) 
		cldice_std = statistics.stdev(cldice_vals)

		with open('./metrics_'+str(k)+'.txt', 'w') as f:
			f.write('DICE METRIC\nPortal mean: {}\nStd: {}\nHepatic mean: {}\nStd: {}\nTotal mean: {}\nStd: {}\nHAUSDORFF DISTANCE\nPortal mean: {}\nStd: {}\nHepatic mean: {}\nStd: {}\nTotal mean: {}\nStd: {}\nAVERAGE SURFACE DISTANCE\nPortal mean: {}\nStd: {}\nHepatic mean: {}\nStd: {}\nTotal mean: {}\nStd: {}\nCLDICE METRIC\nPortal mean: {}\nStd: {}\nHepatic mean: {}\nStd: {}\nTotal mean: {}\nStd: {}\n'.format(
				dice_portal_mean, dice_portal_std,
				dice_hepatic_mean, dice_hepatic_std,
				dice_mean, dice_std,
				haus_portal_mean, haus_portal_std,
				haus_hepatic_mean, haus_hepatic_std,
				haus_mean, haus_std,
				avg_portal_mean, avg_portal_std,
				avg_hepatic_mean, avg_hepatic_std,
				avg_mean, avg_std,
				cldice_portal_mean, cldice_portal_std,
				cldice_hepatic_mean, cldice_hepatic_std,
				cldice_mean, cldice_std
			))

	return [dice_mean, dice_std,
	dice_portal_mean, dice_portal_std,
	dice_hepatic_mean, dice_hepatic_std,
	haus_mean, haus_std,
	haus_portal_mean, haus_portal_std,
	haus_hepatic_mean, haus_hepatic_std,
	avg_mean, avg_std, 
	avg_portal_mean, avg_portal_std,
	avg_hepatic_mean, avg_hepatic_std,
	cldice_mean, cldice_std,
	cldice_portal_mean, cldice_portal_std,
	cldice_hepatic_mean, cldice_hepatic_std]



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
