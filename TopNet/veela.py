from math import prod
import nibabel as nib
import numpy as np
import os
import skimage.transform as skTrans
import skan
from sklearn.cluster import KMeans
import torch
import pandas as pd
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize_3d
from skimage.util import img_as_float32
from scipy.ndimage.morphology import distance_transform_edt as DTM
from monai.transforms import LabelToContour

from tqdm import tqdm


def compute_distance_map(volume):
	# Ensure binary behaviour for skeletonization
	volume[volume != 0] = 1 
	skel = img_as_float32(skeletonize_3d(volume))

	# return np.multiply( (DTM(1-skel) + 1), volume)
	distances, indices = DTM(1 - skel, return_indices=True)
	
	return distances + 1, indices


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
		
		elif '-VE.nii.gz' in name: # INPUT VOLUME
			append_value_to_dict(dict_names, 'Volume shape', nib.load(dataset_path + '/'+ name).shape)
			append_value_to_dict(dict_names, 'Header', nib.load(dataset_path + '/'+ name).header)
			# append_value_to_dict(dict_names, 'Image nifti object', nib.load(dataset_path + '/'+ name))
		
		"""elif '-VE-por.nii.gz' in name: # PORTAL VEINS
			append_value_to_dict(dict_names, 'Portal nifti object', nib.load(dataset_path + '/'+ name))

		elif '-VE-hep.nii.gz' in name: # HEPATIC VEINS
			append_value_to_dict(dict_names, 'Hepatic nifti object', nib.load(dataset_path + '/'+ name))"""

	return dict_names

def cluster_tree(volume, skel, indices):
	# skel = img_as_float32(skeletonize_3d(volume)).astype(np.uint8)
	# _, indices = DTM(1-skel, return_indices=True)
	x,y,z = np.where(volume == 1)
	skeleton = skan.Skeleton(skel)

	clustered_tree = np.zeros_like(skeleton.path_label_image())

	for h,w,d in product(x,y,z):
		clustered_tree[h,w,d] = skeleton.path_label_image()[indices[:,h,w,d][0], indices[:,h,w,d][1], indices[:,h,w,d][2]]

	return clustered_tree


def get_tree_radius(labelled_tree, distances):
	radius_label = np.zeros((len(np.unique(labelled_tree)) - 1, 2)) # -1 to exclude background
	radius_label[:,0] = np.unique(labelled_tree) [1:]
	labelled_tree_t = torch.from_numpy(labelled_tree)
	distances_t = torch.from_numpy(distances)

	for idx, label in enumerate(np.unique(labelled_tree) [1:]): # To directly exclude background
		binary_tree = (labelled_tree_t.type(torch.LongTensor)).where(labelled_tree_t == torch.tensor(label).type(torch.LongTensor), torch.tensor(0.0).type(torch.LongTensor))
		contour = LabelToContour()(binary_tree)
		radius = torch.multiply(distances_t, contour)
		mean_radius_branch = np.mean(radius.numpy() [contour.numpy() == 1])
		radius_label[idx,1] = mean_radius_branch
	
	return radius_label

def create_binary_tree_given_label(labelled_tree, label, binary_tree):

	indices = np.argwhere(labelled_tree == label)

	for idx in indices:
		binary_tree[idx[0], idx[1], idx[2]] = 1
	
	return binary_tree

def kmeans_tree(radius, labelled_tree):
	# for radius in radii:
	low_radio, mid_radio, high_radio = np.zeros_like(labelled_tree), np.zeros_like(labelled_tree), np.zeros_like(labelled_tree)

	km = KMeans(n_clusters=3)
	kmeans = km.fit_predict(radius[:,1].reshape(-1,1))

	idx_lows = radius[:,0] [kmeans == 0]
	idx_meds = radius[:,0] [kmeans == 1]
	idx_highs = radius[:,0] [kmeans == 2]
	
	for label_low in idx_lows:
		low_radio = create_binary_tree_given_label(labelled_tree, label_low, low_radio)
	for label_mid in idx_meds:
		mid_radio = create_binary_tree_given_label(labelled_tree, label_mid, mid_radio)
	for label_high in idx_highs:
		high_radio = create_binary_tree_given_label(labelled_tree, label_high, high_radio)
		


	return low_radio, mid_radio, high_radio

def histogram_PDE_radius(radius_trees):

	dist = pd.DataFrame(radius_trees, colums = ['portal', 'hepatic'])
	dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2)

	fig, ax = plt.subplots()
	dist.plot.kde(ax=ax, legend=False, title='Average radius')
	dist.plot.hist(density=True, ax=ax)
	ax.set_ylabel('Probability')
	ax.set_xlabel('Radius [mm]')
	ax.grid(axis='y')
	ax.set_facecolor('#d8dcd6')


def get_bifurcation_coords(skel):
	skeleton = skan.Skeleton(skel)
	return skeleton.coordinates[skeleton.degrees > 2]	

def compute_orientation_skel(skel):
	skeleton = skan.Skeleton(skel)
	orientations = np.zeros((3, np.shape(skel)[0], np.shape(skel)[1], np.shape(skel)[2]))

	for label in np.unique(skeleton.path_label_image()) [1:]:
		coords_branch_i = skeleton.path_coordinates(index = int(label) - 1)
		for coord_idx, coord in enumerate(coords_branch_i [:-1]):
			vector = coord - coords_branch_i[coord_idx + 1, :]
			vector_norm = vector / np.linalg.norm(vector)
			orientations[:, int(coord[0]), int(coord[1]), int(coord[2])] = vector_norm

	return orientations

def compute_orientation_tree(volume, indices):
	x,y,z = np.where(volume == 1)
	skel = img_as_float32(skeletonize_3d(volume))

	orientation_tree = np.zeros((3, np.shape(skel)[0], np.shape(skel)[1], np.shape(skel)[2]))
	orientation_skel = compute_orientation_skel(skel)

	for h,w,d in product(x,y,z):
		orientation_tree[:,h,w,d] = orientation_skel[:, indices[:,h,w,d][0], indices[:,h,w,d][1], indices[:,h,w,d][2]]

	return orientation_tree
	

def pipeline_1(info_dict, dst_folder, args):

	# PIPELINE (INPUT)
	for idx in tqdm(range(len(info_dict['Image name']))):
		# NIFTI 2 NUMPY ND ARRAY
		ima = nib.load(os.path.join(args.dataset_path, info_dict['Image name'][idx])).get_fdata()
		ima_portal = nib.load(os.path.join(args.dataset_path, info_dict['Portal veins name'][idx])).get_fdata()# .astype(np.uint8)
		ima_hepatic = nib.load(os.path.join(args.dataset_path, info_dict['Hepatic veins name'][idx])).get_fdata()
		
		# 3D INDEXING
		liver = extract_liver(ima, info_dict, idx)
		liver_portal = extract_liver(ima_portal, info_dict, idx)
		liver_hepatic = extract_liver(ima_hepatic, info_dict, idx)

		# RESIZE INPUT AND LABEL (+ LABEL BINARIZATION)
		resized_liver = skTrans.resize(liver, args.input_size, order = 2, preserve_range=True, anti_aliasing = True)

		resized_liver_portal = binarize(skTrans.resize(liver_portal, args.input_size, order = 0, preserve_range=True, anti_aliasing = True))
		resized_liver_portal_dmap, indices_portal = compute_distance_map(resized_liver_portal)
		oriented_portal = compute_orientation_tree(resized_liver_portal, indices_portal)

		resized_liver_hepatic = binarize(skTrans.resize(liver_hepatic, args.input_size, order = 0, preserve_range=True, anti_aliasing = True))
		resized_liver_hepatic_dmap, indices_hepatic = compute_distance_map(resized_liver_hepatic)
		oriented_hepatic = compute_orientation_tree(resized_liver_hepatic, indices_hepatic)

		# SAVE RESIZED IMAGE
		output_ima = nib.Nifti1Image(resized_liver, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
		nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver.nii.gz')

		if args.binary:
			# SAVE RESIZED IMAGE
			
			output_portal = nib.Nifti1Image(resized_liver_portal, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_portal, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_vessel.nii.gz')
			output_portal_dmap = nib.Nifti1Image(resized_liver_portal_dmap, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_portal_dmap, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_dmap.nii.gz')
			output_ima = nib.Nifti1Image(oriented_portal, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_ori.nii.gz')		
			
			output_hepatic = nib.Nifti1Image(resized_liver_hepatic, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_hepatic, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_hep_vessel.nii.gz')
			output_hepatic_dmap = nib.Nifti1Image(resized_liver_hepatic_dmap, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_hepatic_dmap, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_hep_dmap.nii.gz')
			output_ima = nib.Nifti1Image(oriented_hepatic, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_hep_ori.nii.gz')

		else:
			resized_multilabel = np.zeros_like(resized_liver_portal)
			resized_multilabel = resized_liver_portal + binarize(resized_liver_hepatic,2)

			# SAVE RESIZED LABEL
			output_ima = nib.Nifti1Image(resized_multilabel, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_multi_GT.nii.gz')

			resized_multilabel = np.zeros_like(resized_liver_portal_dmap)
			resized_multilabel = resized_liver_portal_dmap + resized_liver_hepatic_dmap

			# SAVE RESIZED DISTANCE MAPS
			output_ima = nib.Nifti1Image(resized_multilabel, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_multi_dmap.nii.gz')

			resized_multilabel = np.zeros_like(oriented_portal)
			resized_multilabel = oriented_portal + oriented_hepatic

			# SAVE RESIZED DISTANCE MAPS
			output_ima = nib.Nifti1Image(resized_multilabel, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_multi_ori.nii.gz')







if __name__ == '__main__':

	dataset_path = '/home/guijosa/Documents/PythonDocs/VEELA/dataset'
	# dataset_path = '/home/guijosa/Documents/PythonDocs/TopNet/data_reshaped_224x224x128_binary'
	radius_p, radius_h = list(), list()

	bar_ = tqdm(sorted(os.listdir(dataset_path)))
	for name in bar_:
		bar_.set_description('Processing {}'.format(name))
		if 'por' in name:
			# Portal
			ima_nifti = nib.load(os.path.join(dataset_path, name))
			ima = ima_nifti.get_fdata()
			skel = img_as_float32(skeletonize_3d(ima)).astype(np.uint8)
			distances, indices = DTM(1 - skel, return_indices=True)
			
			if not os.path.isfile(os.path.join('/home/guijosa/Documents/PythonDocs/VEELA/dataset_labelled', name.split('-')[0] + '-VE-por-labelled.nii.gz')):
				clustered_tree = cluster_tree(ima, skel, indices)
				output_ima = nib.Nifti1Image(clustered_tree, ima_nifti.affine, ima_nifti.header)
				nib.save(output_ima, os.path.join('/home/guijosa/Documents/PythonDocs/VEELA/dataset_labelled', name.split('-')[0] + '-VE-por-labelled.nii.gz'))
			else:
				clustered_tree = nib.load(os.path.join('/home/guijosa/Documents/PythonDocs/VEELA/dataset_labelled', name.split('-')[0] + '-VE-por-labelled.nii.gz')).get_fdata()

			radius = get_tree_radius(clustered_tree, distances)
			low, mid, high = kmeans_tree(radius, clustered_tree)
			output_ima = nib.Nifti1Image(binarize(low) + binarize(mid, 2) + binarize(high,3), ima_nifti.affine, ima_nifti.header)
			nib.save(output_ima, os.path.join('/home/guijosa/Documents/PythonDocs/VEELA/3_clusters', name.split('-')[0] + '-VE-por-segmented.nii.gz'))
			radius_p.append(radius)

		elif 'hep' in name:
			# Labelled hepatic
			ima_nifti = nib.load(os.path.join(dataset_path, name))
			ima = ima_nifti.get_fdata()
			skel = img_as_float32(skeletonize_3d(ima)).astype(np.uint8)
			distances, indices = DTM(1 - skel, return_indices=True)

			if not os.path.isfile(os.path.join('/home/guijosa/Documents/PythonDocs/VEELA/dataset_labelled', name.split('-')[0] + '-VE-hep-labelled.nii.gz')):
				clustered_tree = cluster_tree(ima, skel, indices)
				output_ima = nib.Nifti1Image(clustered_tree, ima_nifti.affine, ima_nifti.header)
				nib.save(output_ima, os.path.join('/home/guijosa/Documents/PythonDocs/VEELA/dataset_labelled', name.split('-')[0] + '-VE-hep-labelled.nii.gz'))
			else:
				clustered_tree = nib.load(os.path.join('/home/guijosa/Documents/PythonDocs/VEELA/dataset_labelled', name.split('-')[0] + '-VE-hep-labelled.nii.gz')).get_fdata()

			radius = get_tree_radius(clustered_tree, distances)
			low, mid, high = kmeans_tree(radius, clustered_tree)
			output_ima = nib.Nifti1Image(binarize(low) + binarize(mid, 2) + binarize(high,3), ima_nifti.affine, ima_nifti.header)
			nib.save(output_ima, os.path.join('/home/guijosa/Documents/PythonDocs/VEELA/3_clusters', name.split('-')[0] + '-VE-hep-segmented.nii.gz'))
			radius_h.append(radius)

	print('DONE')
