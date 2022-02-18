import os 
import json
import numpy as np
import random
import torch
import skimage.transform as skTrans
import nibabel as nib

from medpy.metric.binary import dc as DiceMetric
from tqdm import tqdm
from pathlib import Path
from monai.config import PathLike
from typing import Dict, List
from monai.data import decollate_batch
from monai.transforms import AsDiscrete,  Activations, EnsureType, Compose
from typing import overload

post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dir(path):
	if not  os.path.exists(path):
		os.makedirs(path)



def get_index_dict(my_dict):

	all_list = list()
	for element in my_dict['Image name']:
		all_list.append(element.split('-')[0])

	return all_list

def kfcv(dataset, k):
	print('Performing K-Fold Cross-Vailidation... ')
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
		folds_test[:,fold] = test

	return folds_training.astype(int), folds_validation.astype(int), folds_test.astype(int)

def open_json_file(directory):
	with open(directory) as json_file:
		dictionary = json.load(json_file)
	return dictionary

def save_dataset_dict(info_dict, dataset_name, save_dir):

	# save_dir = 'home2/alberto/data' # '...'
	# Serializing json 
	json_object = json.dumps(info_dict, indent = 4)

	# Writing to sample.json
	with open(os.path.join(save_dir, dataset_name) + '.json', 'w') as outfile:
		outfile.write(json_object)


def create_json_file(dst_folder, info_dict, k): # Add segmentation flag for future

	indexes_list = get_index_dict(info_dict)
	folds_training, folds_validation, folds_test = kfcv(indexes_list, k)
	json_routes = list()
	dictionary_list = list()

	for fold in tqdm(range(np.shape(folds_training)[1])):
		# Data to be written
		dictionary = {
			"description": "btcv yucheng",
			"labels": {
				"0": "background",
				"1": "portal_vessels"
				},
			"licence": "yt",
			"modality": {
				"0": "CT"
				},
			"name": "VEELA",
			"numTest": 20,
			"numTraining": 80,
			"reference": "Vanderbilt University",
			"tensorImageSize": "3D",
			"test": [
			{
				"image": os.path.join(dst_folder,str(folds_test[0, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_test[0,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_test[1, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_test[1,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_test[2, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_test[2,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_test[3, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_test[3,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_test[4, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_test[4,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_test[5, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_test[5,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_test[6, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_test[6,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			],
			"training": [
			{
				"image": os.path.join(dst_folder,str(folds_training[0, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[0,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[1, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[1,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[2, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[2,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[3, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[3,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[4, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[4,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[5, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[5,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[6, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[6,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[7, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[7,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[8, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[8,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[9, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[9,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[10, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[10,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[11, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[11,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[12, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[12,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[13, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[13,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[14, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[14,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[15, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[15,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[16, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[16,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[17, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[17,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[18, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[18,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[19, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[19,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[20, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[20,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[21, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[21,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_training[22, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_training[22,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			],
			"validation": [
			{
				"image": os.path.join(dst_folder,str(folds_validation[0, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_validation[0,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_validation[1, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_validation[1,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_validation[2, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_validation[2,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_validation[3, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_validation[3,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			{
				"image": os.path.join(dst_folder,str(folds_validation[4, fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				"label": os.path.join(dst_folder,str(folds_validation[4,fold]).zfill(3) + '-VE-liver_por_GT' + '.nii.gz')
			},
			]
		}
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


def save_segmentations(model,weights_dir, json_dict, info_dict, size, test_loader):
	output_route = os.path.join(os.path.abspath(os.getcwd()), 'results')
	create_dir(output_route)
	model.load_state_dict(torch.load(os.path.join(weights_dir, "best_metric_model.pth")))
	model.eval()
	with torch.no_grad():
		for idx, test_data in enumerate(test_loader):
			val_images, val_labels = test_data["image"].to(device), test_data["label"].to(device)
			pred = model(val_images)
			pred = post_trans(decollate_batch(pred))

			result = pred[0].squeeze().cpu().numpy().astype(np.uint8)

			pos = get_position(info_dict, 'Image name', json_dict['test'][idx]['image'].split('/')[-1])
			unresized_result = skTrans.resize(result, (
				info_dict['Liver coordinates'][pos][1] - info_dict['Liver coordinates'][pos][0] + 1,
				info_dict['Liver coordinates'][pos][3] - info_dict['Liver coordinates'][pos][2] + 1,
				info_dict['Liver coordinates'][pos][5] - info_dict['Liver coordinates'][pos][4] + 1 
			), preserve_range=True).astype(np.uint8)

			result = np.zeros(info_dict['Volume shape'][pos]).astype(np.uint8)
			result[
				info_dict['Liver coordinates'][pos][0]:info_dict['Liver coordinates'][pos][1] + 1,
				info_dict['Liver coordinates'][pos][2]:info_dict['Liver coordinates'][pos][3] + 1,
				info_dict['Liver coordinates'][pos][4]:info_dict['Liver coordinates'][pos][5] + 1 
			] = unresized_result
			output_ima = nib.Nifti1Image(result, info_dict['Affine matrix'][pos], info_dict['Header'][pos])

			groundtruth = info_dict['Portal nifti object'][pos].get_fdata()
			# Compute dice metric
			dice = 100.*DiceMetric(result.astype(np.bool), groundtruth.astype(np.bool))
			print('Dice metric: {} %'.format(dice))

			nib.save(output_ima, output_route + '/' + info_dict['Image name'][pos] + '_segmented.nii.gz')


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






