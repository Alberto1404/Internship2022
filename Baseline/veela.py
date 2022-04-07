import nibabel as nib
import numpy as np
import os
import skimage.transform as skTrans

from tqdm import tqdm

def binarize(volume, value = 1):
	volume[volume > 0.95] = value
	volume[volume != value] = 0

	return volume


def resize(input_toresize, info_dict, idx):

	output_resized = skTrans.resize(input_toresize, (
		info_dict['Liver coordinates'][idx][1] - info_dict['Liver coordinates'][idx][0] + 1,
		info_dict['Liver coordinates'][idx][3] - info_dict['Liver coordinates'][idx][2] + 1,
		info_dict['Liver coordinates'][idx][5] - info_dict['Liver coordinates'][idx][4] + 1 
	), preserve_range=True, order = 0, anti_aliasing=True)

	return output_resized

def index_liver_in_volume(liver, info_dict, idx):

	result = np.zeros(info_dict['Volume shape'][idx])
	result[
		info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1] + 1,
		info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3] + 1,
		info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5] + 1 
	] = liver
	return result

def extract_liver(volume_image, info_dict, image_idx):
	liver = volume_image[
			info_dict['Liver coordinates'][image_idx][0]:info_dict['Liver coordinates'][image_idx][1] + 1,
			info_dict['Liver coordinates'][image_idx][2]:info_dict['Liver coordinates'][image_idx][3] + 1,
			info_dict['Liver coordinates'][image_idx][4]:info_dict['Liver coordinates'][image_idx][5] + 1
	]
	return liver


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

def get_names_from_dataset(dataset_path, dataset_name, save_dir):

	dict_names = dict()

	for name in tqdm(sorted(os.listdir(dataset_path))):
		if '-VE.nii.gz' in name: # INPUT VOLUME
			append_value_to_dict(dict_names, 'Image name', name)
		elif '-VE-por.nii.gz' in name: # PORTAL VEINS
			append_value_to_dict(dict_names, 'Portal veins name', name)
		elif '-VE-hep.nii.gz' in name: # HEPATIC VEINS
			append_value_to_dict(dict_names, 'Hepatic veins name', name)


	return dict_names
		

def load_dataset(dict_names, dataset_path):

	for name in tqdm(sorted(os.listdir(dataset_path))):
		if '-VE-liv.nii.gz' in name: # LIVER MASKS
			liver_nifti = nib.load(dataset_path + '/' + name)
			liver_coords = get_liver_bounding_box(liver_nifti.get_fdata())
			append_value_to_dict(dict_names, 'Liver coordinates', liver_coords)
			append_value_to_dict(dict_names, 'Affine matrix', liver_nifti.affine)
			# QUITAR LUEGO append_value_to_dict(dict_names, 'Liver mask name', name)
		
		elif '-VE.nii.gz' in name: # INPUT VOLUME
			append_value_to_dict(dict_names, 'Volume shape', nib.load(dataset_path + '/'+ name).shape)
			append_value_to_dict(dict_names, 'Header', nib.load(dataset_path + '/'+ name).header)
			# append_value_to_dict(dict_names, 'Image nifti object', nib.load(dataset_path + '/'+ name))
		
		"""elif '-VE-por.nii.gz' in name: # PORTAL VEINS
			append_value_to_dict(dict_names, 'Portal nifti object', nib.load(dataset_path + '/'+ name))

		elif '-VE-hep.nii.gz' in name: # HEPATIC VEINS
			append_value_to_dict(dict_names, 'Hepatic nifti object', nib.load(dataset_path + '/'+ name))"""

	return dict_names


def pipeline_1(info_dict, dst_folder, args):

	# PIPELINE (INPUT)
	for idx in tqdm(range(len(info_dict['Image name']))):
		# NIFTI 2 NUMPY ND ARRAY
		ima = nib.load(os.path.join(args.dataset_path, info_dict['Image name'][idx])).get_fdata()
		ima_portal = nib.load(os.path.join(args.dataset_path, info_dict['Portal veins name'][idx])).get_fdata()# .astype(np.uint8)
		# QUITAR LUEGO ima_portal = nib.load(os.path.join(args.dataset_path, info_dict['Liver mask name'][idx])).get_fdata()# .astype(np.uint8)

		# 3D INDEXING
		liver = extract_liver(ima, info_dict, idx)
		liver_portal = extract_liver(ima_portal, info_dict, idx)

		# RESIZE INPUT AND LABEL (+ LABEL BINARIZATION)
		resized_liver = skTrans.resize(liver, args.input_size, order = 1, preserve_range=True, anti_aliasing = True)
		resized_liver_portal = binarize(skTrans.resize(liver_portal, args.input_size, order = 0, preserve_range=True, anti_aliasing = True))

		# SAVE RESIZED IMAGE
		output_ima = nib.Nifti1Image(resized_liver, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
		nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver.nii.gz')

		if not args.binary:
			# NIFTI 2 NUMPY ND ARRAY
			ima_hepatic = nib.load(os.path.join(args.dataset_path, info_dict['Hepatic veins name'][idx])).get_fdata()

			# 3D INDEXING
			liver_hepatic = extract_liver(ima_hepatic, info_dict, idx)

			# RESIZE LABEL (+ LABEL BINARIZATION)
			resized_liver_hepatic = binarize(skTrans.resize(liver_hepatic, args.input_size, order = 0, preserve_range=True, anti_aliasing = True),2)

			resized_multilabel = np.zeros_like(ima_hepatic)
			resized_multilabel = resized_liver_portal + resized_liver_hepatic

			# SAVE RESIZED LABEL
			output_ima = nib.Nifti1Image(resized_multilabel, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_multi_GT.nii.gz')

		else:
			# SAVE RESIZED LABEL		
			output_portal = nib.Nifti1Image(resized_liver_portal, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_portal, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_GT.nii.gz')