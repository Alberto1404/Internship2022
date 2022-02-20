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
			append_value_to_dict(dict_names, 'Image nifti object', nib.load(dataset_path + '/'+ name))
		
		elif '-VE-por.nii.gz' in name: # PORTAL VEINS
			append_value_to_dict(dict_names, 'Portal nifti object', nib.load(dataset_path + '/'+ name))

		elif '-VE-hep.nii.gz' in name: # HEPATIC VEINS
			append_value_to_dict(dict_names, 'Hepatic nifti object', nib.load(dataset_path + '/'+ name))

	return dict_names


def split_dataset(info_dict, size, dst_folder, is_binary):

	# PIPELINE (INPUT)
	for idx in tqdm(range(len(info_dict['Image name']))):
		# NIFTI 2 NUMPY ND ARRAY
		ima = info_dict['Image nifti object'][idx].get_fdata()
		ima_portal = info_dict['Portal nifti object'][idx].get_fdata()
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
		resized_liver = skTrans.resize(liver, size, order = 1, preserve_range=True, anti_aliasing = True)
		resized_liver_portal = skTrans.resize(liver_portal.astype(np.uint8), size, order = 0, preserve_range=True, anti_aliasing = True).astype(np.uint8)

		resized_liver_portal[np.where(resized_liver_portal > 0.95)] = 1
		resized_liver_portal[np.where(resized_liver_portal != 1)] = 0

		# SAVE RESIZED IMAGE
		output_ima = nib.Nifti1Image(resized_liver, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
		nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver.nii.gz')

		if not is_binary:
			ima_hepatic = info_dict['Hepatic nifti object'][idx].get_fdata().astype(np.uint8)
			liver_hepatic = ima_hepatic[
					info_dict['Liver coordinates'][idx][0]:info_dict['Liver coordinates'][idx][1] + 1,
					info_dict['Liver coordinates'][idx][2]:info_dict['Liver coordinates'][idx][3] + 1,
					info_dict['Liver coordinates'][idx][4]:info_dict['Liver coordinates'][idx][5] + 1
				]
			resized_liver_hepatic = skTrans.resize(liver_hepatic.astype(np.uint8), size, order = 0, preserve_range=True, anti_aliasing = True).astype(np.uint8)
			resized_liver_hepatic[np.where(resized_liver_hepatic > 0.95)] = 1
			resized_liver_hepatic[np.where(resized_liver_hepatic != 1)] = 0

			resized_multilabel = np.zeros((2,size[0],size[1],size[2]), dtype=np.uint8)

			resized_multilabel[0,:,:,:] = resized_liver_portal
			resized_multilabel[1,:,:,:] = resized_liver_hepatic

			output_ima = nib.Nifti1Image(resized_multilabel, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_multi_GT.nii.gz')
		
		output_portal = nib.Nifti1Image(resized_liver_portal, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
		nib.save(output_portal, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_GT.nii.gz')

	"""# PIPELINE
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
		nib.save(output_ima_gt, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_GT.nii.gz')"""



