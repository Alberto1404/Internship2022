import os 
import json
import numpy as np
import random
import torch
import skimage.transform as skTrans
import nibabel as nib
from tqdm import tqdm

from veela import process_dataset
from monai.inferers import sliding_window_inference


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
			# any(item in a for item in b)
			validation = sorted(random.sample(dataset,5))
		print('\nValidation indexes obtained!!!')
		test = sorted(list(set(dataset) - set(training) - set(validation)))

		folds_training[:,fold] = training
		folds_validation[:,fold] = validation
		folds_test[:,fold] = test

	return folds_training.astype(int), folds_validation.astype(int), folds_test.astype(int)


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
				os.path.join(dst_folder,str(folds_test[0,fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				os.path.join(dst_folder,str(folds_test[1,fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				os.path.join(dst_folder,str(folds_test[2,fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				os.path.join(dst_folder,str(folds_test[3,fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				os.path.join(dst_folder,str(folds_test[4,fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				os.path.join(dst_folder,str(folds_test[5,fold]).zfill(3) + '-VE-liver' + '.nii.gz'),
				os.path.join(dst_folder,str(folds_test[6,fold]).zfill(3) + '-VE-liver' + '.nii.gz')
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


def save_segmentations(model,weights_dir, json_dict, info_dict, size, val_ds):
	model.load_state_dict(torch.load(os.path.join(weights_dir, "best_metric_model.pth")))
	# model.load_state_dict(torch.load('./best_metric_model.pth')) # PRETRAINED NETWORK
	model.eval()
	for case_num in range(len(json_dict['validation'])): # len(dictionary['validation']) defined in JSON !!!
		with torch.no_grad():
			img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
			img = val_ds[case_num]["image"]
			label = val_ds[case_num]["label"]
			val_inputs = torch.unsqueeze(img, 1).cuda()
			val_labels = torch.unsqueeze(label, 1).cuda()
			val_outputs = sliding_window_inference(
			val_inputs, size, 4, model, overlap=0.8
			)
			# result = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, :].permute(2,0,1).numpy()
			result = torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, :].numpy()


			pos = get_position(info_dict, 'Image name', img_name)

			unresized_result = skTrans.resize(result, (
			info_dict['Liver coordinates'][pos][1] - info_dict['Liver coordinates'][pos][0] + 1,
			info_dict['Liver coordinates'][pos][3] - info_dict['Liver coordinates'][pos][2] + 1,
			info_dict['Liver coordinates'][pos][5] - info_dict['Liver coordinates'][pos][4] + 1 
			), order = 1, preserve_range=True)

			result = np.zeros(info_dict['Volume shape'][pos])
			result[
			info_dict['Liver coordinates'][pos][0]:info_dict['Liver coordinates'][pos][1] + 1,
			info_dict['Liver coordinates'][pos][2]:info_dict['Liver coordinates'][pos][3] + 1,
			info_dict['Liver coordinates'][pos][4]:info_dict['Liver coordinates'][pos][5] + 1 
			] = unresized_result
			output_ima = nib.Nifti1Image(result, info_dict['Affine matrix'][pos], info_dict['Header'][pos])
			# output_route = './results'
			output_route = os.path.join(os.path.abspath(os.getcwd()), 'results')
			create_dir(output_route)
			nib.save(output_ima, output_route + '/' + info_dict['Image name'][pos] + '_segmented.nii.gz')

		





