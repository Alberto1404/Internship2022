import nibabel as nib
import numpy as np
import os
import skimage.transform as skTrans
from tqdm import tqdm


def get_liver_bounding_box(liver):

	X, Y, Z = np.where(liver > 0)
	return np.array([np.min(X), np.max(X), np.min(Y), np.max(Y), np.min(Z), np.max(Z)])

def append_value_to_dict(dict_obj, key, value):

	# Check if key exist in dict or not
	if key in dict_obj:
		# Key exist in dict.
		# Check if type of value of key is list or not
		if not isinstance(dict_obj[key], list):
			# If type is not list then make it list
			dict_obj[key] = [dict_obj[key]]
		# Append the value in list
		dict_obj[key].append(value)
	else:
		# As key is not in dict,
		# so, add key-value pair
		dict_obj[key] = value


def process_dataset(dataset_path):

	info_dict = dict()

	for name in tqdm(sorted(os.listdir(dataset_path))):
		if '-VE-liv.nii.gz' in name: # LIVER MASKS
			liver_nifti = nib.load(dataset_path + '/'+name)
			liver_coords = get_liver_bounding_box(liver_nifti.get_fdata())
			append_value_to_dict(info_dict, 'Liver coordinates', liver_coords)
			append_value_to_dict(info_dict, 'Affine matrix', liver_nifti.affine)
		
		elif '-VE.nii.gz' in name: # INPUT VOLUME
			append_value_to_dict(info_dict, 'Image name', name)
			append_value_to_dict(info_dict, 'Volume shape', nib.load(dataset_path + '/'+ name).shape)
			append_value_to_dict(info_dict, 'Header', nib.load(dataset_path + '/'+ name).header)
			append_value_to_dict(info_dict, 'Image nifti object', nib.load(dataset_path + '/'+ name))
		
		elif '-VE-por.nii.gz' in name: # PORTAL VEINS
			append_value_to_dict(info_dict, 'Portal veins name', name)
			append_value_to_dict(info_dict, 'Portal nifti object', nib.load(dataset_path + '/'+ name))
		elif '-VE-hep.nii.gz' in name: # HEPATIC VEINS
			append_value_to_dict(info_dict, 'Hepatic veins name', name)
			append_value_to_dict(info_dict, 'Hepatic nifti object', nib.load(dataset_path + '/'+ name))

	return info_dict

def split_dataset(info_dict, dataset_path, size, dst_folder):

	# PIPELINE
	for idx in tqdm(range(len(info_dict['Image name']))):
		name = info_dict['Image name'][idx]
		name_gt = info_dict['Portal veins name'][idx]
		# NIFTI 2 NUMPY ND ARRAY
		ima = nib.load(os.path.join(dataset_path, name)).get_fdata()
		ima_gt = nib.load(os.path.join(dataset_path, name_gt)).get_fdata().astype(np.uint8)
		# 3D indexing
		liver = ima[
			info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1] + 1,
			info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3] + 1,
			info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5] + 1
		]
		liver_gt = ima_gt[
			info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1] + 1,
			info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3] + 1,
			info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5] + 1
		]
		# RESIZE
		resized_liver = skTrans.resize(liver, size, order = 1, preserve_range=True)
		resized_liver_gt = skTrans.resize(liver_gt, size, preserve_range=True)

		resized_liver_gt[np.where(resized_liver_gt > 0.95)] = 1
		resized_liver_gt[np.where(resized_liver_gt != 1)] = 0

		output_ima = nib.Nifti1Image(resized_liver, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
		output_ima_gt = nib.Nifti1Image(resized_liver_gt, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
		nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver.nii.gz')
		nib.save(output_ima_gt, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_GT.nii.gz')

	"""for idx, name in enumerate(info_dict['Image name']):
		ima = nib.load(os.path.join(dataset_path, name)).get_fdata()
		# 3D indexing volume_images
		liver = ima[
			info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1] + 1,
			info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3] + 1,
			info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5] + 1
		]
		# resized_liver = skTrans.resize(liver, (average_size[0], average_size[1], average_size[2]), order = 1, preserve_range=True)
		resized_liver = skTrans.resize(liver, size, order = 1, preserve_range=True)
		output_ima = nib.Nifti1Image(resized_liver, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
		if len(os.listdir('./data/imagesTr')) < 28:
			nib.save(output_ima, './data/imagesTr/' + info_dict['Image name'][idx].split('.')[0] + '-liver.nii.gz')
		else:
			nib.save(output_ima, './data/imagesTs/' + info_dict['Image name'][idx].split('.')[0] + '-liver.nii.gz')

	for idx, name in enumerate(info_dict['Portal veins name']):
		ima = nib.load(os.path.join(dataset_path, name)).get_fdata().astype(np.uint8)
		# 3D indexing
		liver = ima[
			info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1] + 1,
			info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3] + 1,
			info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5] + 1
		]
		# resized_liver = skTrans.resize(liver, (average_size[0], average_size[1], average_size[2]), order = 1, preserve_range=True)
		resized_liver = skTrans.resize(liver, size, preserve_range=True)

		resized_liver[np.where(resized_liver > 0.95)] = 1
		resized_liver[np.where(resized_liver != 1)] = 0

		output_ima = nib.Nifti1Image(resized_liver, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
		if len(os.listdir('./data/labelsTr')) < 28:
			nib.save(output_ima, './data/labelsTr/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_GT.nii.gz')
		else:
			nib.save(output_ima, './data/labelsTs/'+ info_dict['Image name'][idx].split('.')[0] + '-liver_por_GT.nii.gz')"""


