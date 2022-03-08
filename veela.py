import nibabel as nib
import numpy as np
import os
import skimage.transform as skTrans

from tqdm import tqdm

def binarize(volume):
	volume[volume > 0.95] = 1
	volume[volume != 1] = 0

	return volume


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
		

def process_dataset(dict_names, dataset_path):

	for name in tqdm(sorted(os.listdir(dataset_path))):
		if '-VE-liv.nii.gz' in name: # LIVER MASKS
			liver_nifti = nib.load(dataset_path + '/' + name)
			liver_coords = get_liver_bounding_box(liver_nifti.get_fdata())
			append_value_to_dict(dict_names, 'Liver coordinates', liver_coords)
			append_value_to_dict(dict_names, 'Affine matrix', liver_nifti.affine)
		
		elif '-VE.nii.gz' in name: # INPUT VOLUME
			append_value_to_dict(dict_names, 'Volume shape', nib.load(dataset_path + '/'+ name).shape)
			append_value_to_dict(dict_names, 'Header', nib.load(dataset_path + '/'+ name).header)
			# append_value_to_dict(dict_names, 'Image nifti object', nib.load(dataset_path + '/'+ name))
		
		"""elif '-VE-por.nii.gz' in name: # PORTAL VEINS
			append_value_to_dict(dict_names, 'Portal nifti object', nib.load(dataset_path + '/'+ name))

		elif '-VE-hep.nii.gz' in name: # HEPATIC VEINS
			append_value_to_dict(dict_names, 'Hepatic nifti object', nib.load(dataset_path + '/'+ name))"""

	return dict_names


def split_dataset(info_dict, dst_folder, args):

	# PIPELINE (INPUT)
	for idx in tqdm(range(len(info_dict['Image name']))):
		# NIFTI 2 NUMPY ND ARRAY
		ima = nib.load(os.path.join(args.dataset_path, info_dict['Image name'][idx])).get_fdata()
		ima_portal = nib.load(os.path.join(args.dataset_path, info_dict['Portal veins name'][idx])).get_fdata()# .astype(np.uint8)
		# 3D INDEXING
		liver = ima[
			info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1] + 1,
			info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3] + 1,
			info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5] + 1
		]
		liver_portal = ima_portal[
			info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1] + 1,
			info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3] + 1,
			info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5] + 1
		]
		# RESIZE
		resized_liver = skTrans.resize(liver, args.input_size, order = 1, preserve_range=True, anti_aliasing = True)
		resized_liver_portal = skTrans.resize(liver_portal, args.input_size, order = 0, preserve_range=True, anti_aliasing = True)

		resized_liver_portal = binarize(resized_liver_portal)
		# resized_liver_portal[np.where(resized_liver_portal > 0.95)] = 1
		# resized_liver_portal[np.where(resized_liver_portal != 1)] = 0

		# SAVE RESIZED IMAGE
		output_ima = nib.Nifti1Image(resized_liver, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
		nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver.nii.gz')

		if not args.binary:
			ima_hepatic = nib.load(os.path.join(args.dataset_path, info_dict['Hepatic veins name'][idx])).get_fdata()
			liver_hepatic = ima_hepatic[
					info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1] + 1,
					info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3] + 1,
					info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5] + 1
				]
			resized_liver_hepatic = skTrans.resize(liver_hepatic, args.input_size, order = 0, preserve_range=True, anti_aliasing = True)
			resized_liver_hepatic = binarize(resized_liver_hepatic)
			# resized_liver_hepatic[np.where(resized_liver_hepatic > 0.95)] = 1
			# resized_liver_hepatic[np.where(resized_liver_hepatic != 1)] = 0
			resized_liver_hepatic[np.where(resized_liver_hepatic == 1)] = 2 # Assign new label

			resized_multilabel = np.zeros_like(ima_hepatic)
			resized_multilabel = resized_liver_portal + resized_liver_hepatic

			output_ima = nib.Nifti1Image(resized_multilabel, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_multi_GT.nii.gz')

			"""ima_hepatic = nib.load(os.path.join(args.dataset_path, info_dict['Hepatic veins name'][idx])).get_fdata()
			liver_hepatic = ima_hepatic[
					info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1] + 1,
					info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3] + 1,
					info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5] + 1
				]
			resized_liver_hepatic = skTrans.resize(liver_hepatic, args.input_size, order = 0, preserve_range=True, anti_aliasing = True)
			resized_liver_hepatic[np.where(resized_liver_hepatic > 0.95)] = 1
			resized_liver_hepatic[np.where(resized_liver_hepatic != 1)] = 0

			resized_multilabel = np.zeros((2,args.input_size[0],args.input_size[1],args.input_size[2]))

			resized_multilabel[0,:,:,:] = resized_liver_portal
			resized_multilabel[1,:,:,:] = resized_liver_hepatic

			output_ima = nib.Nifti1Image(resized_multilabel, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_multi_GT.nii.gz')"""
		else:		
			output_portal = nib.Nifti1Image(resized_liver_portal, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_portal, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_GT.nii.gz')