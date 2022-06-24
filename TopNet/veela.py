from math import dist
import nibabel as nib
import numpy as np
import os
import skimage.transform as skTrans
import skan
from sklearn.cluster import KMeans
import torch
import pandas as pd
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize_3d, binary_dilation
from skimage.util import img_as_float32
from scipy.ndimage.morphology import distance_transform_edt as DTM
from monai.transforms import LabelToContour, KeepLargestConnectedComponent
import dijkstra3d

from tqdm import tqdm

# CenterLine Dice scores from 
# https://github.com/jocpae/clDice

def cl_score(v, s):
	"""[this function computes the skeleton volume overlap]
	Args:
		v ([bool]): [image]
		s ([bool]): [skeleton]
	Returns:
		[float]: [computed skeleton volume intersection]
	"""
	return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
	"""[this function computes the cldice metric]
	Args:
		v_p ([bool]): [predicted image]
		v_l ([bool]): [ground truth image]
	Returns:
		[float]: [cldice metric]
	"""
	tprec = cl_score(v_p,skeletonize_3d(v_l))
	tsens = cl_score(v_l,skeletonize_3d(v_p))

	result = 2*tprec*tsens/(tprec+tsens)

	return 0 if np.isnan(result) else result


def compute_distance_map(volume):
	# Computation of Centerness score map GT for D2
	
	# Ensure binary behaviour for skeletonization
	volume[volume != 0] = 1 
	skel = img_as_float32(skeletonize_3d(volume))

	distances, indices = DTM(1 - skel, return_indices=True)
	
	return np.multiply(distances, volume), indices, skel


def binarize(volume, value = 1):
	# Ensure masks are binarized after being resized. 
	
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
	
	# Get liver coordinates, as we will work only with this as whole image to reduce data-imbalance

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
	
	# Compute labelled tree mask

	x,y,z = np.where(volume == 1)
	skeleton = skan.Skeleton(skel)

	clustered_tree = np.zeros_like(skeleton.path_label_image())

	for h,w,d in tqdm(zip(x,y,z), desc='Clustering'):
		clustered_tree[h,w,d] = skeleton.path_label_image()[indices[:,h,w,d][0], indices[:,h,w,d][1], indices[:,h,w,d][2]]

	return clustered_tree


def get_tree_radius(labelled_tree, distances):
	
	# Radii estimation, as dot product of centerness score and contour of label (assumption of vasculature as tubular structures)
	
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
	
	# Segment a tree knowing radius, for segmantic segmentation in low-mid-hgih radii vessels (FUTURE WORK)
	
	# for radius in radii:
	low_radio, mid_radio, high_radio = np.zeros_like(labelled_tree), np.zeros_like(labelled_tree), np.zeros_like(labelled_tree)

	km = KMeans(n_clusters=3) # Modify as desired
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
	
	# Fancy representation of radii PDE

	dist = pd.DataFrame(radius_trees, colums = ['portal', 'hepatic'])
	dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2)

	fig, ax = plt.subplots()
	dist.plot.kde(ax=ax, legend=False, title='Average radius')
	dist.plot.hist(density=True, ax=ax)
	ax.set_ylabel('Probability')
	ax.set_xlabel('Radius [mm]')
	ax.grid(axis='y')
	ax.set_facecolor('#d8dcd6')

# Python3 code for generating points on a 3-D line
# using Bresenham's Algorithm

def bresenham3D(A,B):
	# Bresenham algorithm in 3D. Ideally used to enlarge skeleton for 
	
	x1, y1, z1 = A
	x2, y2, z2 = B
	ListOfPoints = []
	ListOfPoints.append((x1, y1, z1))
	dx = abs(x2 - x1)
	dy = abs(y2 - y1)
	dz = abs(z2 - z1)
	
	if (x2 > x1):
		xs = 1
	else:
		xs = -1
	if (y2 > y1):
		ys = 1
	else:
		ys = -1
	if (z2 > z1):
		zs = 1
	else:
		zs = -1

	# Driving axis is X-axis"
	if (dx >= dy and dx >= dz):		
		p1 = 2 * dy - dx
		p2 = 2 * dz - dx
		while (x1 != x2):
			x1 += xs
			if (p1 >= 0):
				y1 += ys
				p1 -= 2 * dx
			if (p2 >= 0):
				z1 += zs
				p2 -= 2 * dx
			p1 += 2 * dy
			p2 += 2 * dz
			ListOfPoints.append((x1, y1, z1))

	# Driving axis is Y-axis"
	elif (dy >= dx and dy >= dz):	
		p1 = 2 * dx - dy
		p2 = 2 * dz - dy
		while (y1 != y2):
			y1 += ys
			if (p1 >= 0):
				x1 += xs
				p1 -= 2 * dy
			if (p2 >= 0):
				z1 += zs
				p2 -= 2 * dy
			p1 += 2 * dx
			p2 += 2 * dz
			ListOfPoints.append((x1, y1, z1))

	# Driving axis is Z-axis"
	else:		
		p1 = 2 * dy - dz
		p2 = 2 * dx - dz
		while (z1 != z2):
			z1 += zs
			if (p1 >= 0):
				y1 += ys
				p1 -= 2 * dz
			if (p2 >= 0):
				x1 += xs
				p2 -= 2 * dz
			p1 += 2 * dy
			p2 += 2 * dx
			ListOfPoints.append((x1, y1, z1))
	return ListOfPoints



def get_bifurcation_coords(skel):
	skeleton = skan.Skeleton(skel)
	return skeleton.coordinates[skeleton.degrees > 2]

def ensure_is_root(skel, binary_tree, labelled_tree, edges_init, root_idx, root):
	dist_list = list()
	for coord in edges_init:
		dist_list.append(np.linalg.norm(root - coord))
	
	closest = edges_init[np.argmin(dist_list)]
	extra_skel = bresenham3D(root.astype(int), closest.astype(int))
	aux_skel = skel.copy()
	for coord in extra_skel:
		aux_skel[coord] = 1

	### CON NUEVO ESQULETO
	aux_skeleton = skan.Skeleton(aux_skel) # aux_skeleton
	edges = skan.Skeleton(skel).coordinates[skan.Skeleton(skel).degrees == 1]
	aux_edges = aux_skeleton.coordinates[aux_skeleton.degrees == 1] # aux_edges
	label_edges= [labelled_tree [int(edges[i][0]), int(edges[i][1]), int(edges[i][2])] for i in range(edges.shape[0])]
	aux_label_edges = [labelled_tree [int(aux_edges[i][0]), int(aux_edges[i][1]), int(aux_edges[i][2])] for i in range(aux_edges.shape[0])]
 
	tree_clusters = np.zeros_like(labelled_tree)
	for coordinate in aux_edges:
		label = labelled_tree[int(coordinate[0]), int(coordinate[1]), int(coordinate[2])]
		tree_clusters[labelled_tree == label] = label
 
	radius_label = get_tree_radius(tree_clusters, np.multiply(DTM(1-aux_skel), binary_tree ))
	new_root_idx = int(radius_label[np.argmax(radius_label[:,1]),0])

	if root_idx != new_root_idx:
		return(ensure_is_root(skel = aux_skel, binary_tree = binary_tree, labelled_tree=labelled_tree, root_idx=new_root_idx, root = edges[np.argwhere(np.array(aux_label_edges) == new_root_idx).item()]))
	else:
		print('Ensured root: {}'.format(edges[np.argwhere(np.array(label_edges) == new_root_idx).item()]))
		# return edges, skel, edges[np.argwhere(np.array(label_edges) == root_idx).item()]
		return aux_edges, aux_skel, edges[np.argwhere(np.array(label_edges) == new_root_idx).item()]

	"""while root_idx != new_root_idx:
		ensure_is_root(skel, DTM(1-skel), labelled_tree,new_root_idx, edges[np.argwhere(np.array(label_edges) == new_root_idx).item()])
	print('Root correcto: {}'.format(edges[np.argwhere(np.array(label_edges) == new_root_idx).item()]))
	return edges, skel, edges[np.argwhere(np.array(label_edges) == new_root_idx).item()]"""
	# oot = edges[np.argwhere(np.array(label_edges) == root_idx).item()]



def compute_orientation_skel(skel, binary_tree, labelled_tree):
	# Computation of orientation vector field GT in D3 (skeleton)

	### 1. Fastest (Orientations collapse in bifurcations, adjacent orientations)
	# skeleton = skan.Skeleton(skel)
	# orientations = np.zeros((3, np.shape(skel)[0], np.shape(skel)[1], np.shape(skel)[2]))

	# for label in np.unique(skeleton.path_label_image()) [1:]:
	# 	coords_branch_i = skeleton.path_coordinates(index = int(label) - 1)
	# 	for coord_idx, coord in enumerate(coords_branch_i [:-1]):
	# 		vector = coord - coords_branch_i[coord_idx + 1, :]
	# 		vector_norm = vector / np.linalg.norm(vector)
	# 		orientations[:, int(coord[0]), int(coord[1]), int(coord[2])] = vector_norm
 
	# return orientations"""

	### 2. ORIENTATION PRESERVED. (New problem: non-connected components return incorrect radii, with its concerning wrong root finding)
	"""orientations = np.zeros((3, np.shape(skel)[0], np.shape(skel)[1], np.shape(skel)[2]))
	edges_init = skan.Skeleton(skel).coordinates[skan.Skeleton(skel).degrees == 1] # EDGES SKEL_INIT
	skel_dilated = img_as_float32(skeletonize_3d(binary_dilation(binary_dilation(skel).astype(np.float32)).astype(np.float32)))
	
	skeleton = skan.Skeleton(skel_dilated)
	edges = skeleton.coordinates[skeleton.degrees == 1]
	# SOLO GUARDAR COORDENADAS DE ESQUELETO MALO
	# cogemos esqueleto malo y lo dilatamos 2 veces, y hacemos todo lo de antes, + nuevo dmap con esqueleto dilatado
	label_edges = [labelled_tree [int(edges[i][0]), int(edges[i][1]), int(edges[i][2])] for i in range(edges.shape[0])]
 
	tree_clusters = np.zeros_like(labelled_tree)
	for coordinate in edges:
		label = labelled_tree[int(coordinate[0]), int(coordinate[1]), int(coordinate[2])]
		tree_clusters[labelled_tree == label] = label
	
	distances = np.multiply(DTM(1-skel_dilated), binary_tree)
	radius_label = get_tree_radius(tree_clusters, distances)
	root_idx = int(radius_label[np.argmax(radius_label[:,1]),0])

	root = edges[np.argwhere(np.array(label_edges) == root_idx).item()]
	print('Root supuestamente correcto: {}'.format(root))

	edges, skel, root = ensure_is_root(skel_dilated, binary_tree, labelled_tree, edges_init, root_idx, root)"""	


	"""try:
		root = edges[np.argwhere(np.array(label_edges) == root_idx).item()]
		print('Root supuestamente correcto: {}'.format(root))
	except ValueError:
		# root = [ edges[np.argwhere(np.array(label_edges) == root_idx)][i][0] for i in range(len(edges[np.argwhere(np.array(label_edges) == root_idx)])) ]
		print('Error!!')"""


	# root = edges[np.argwhere(np.array(label_edges) == root_idx).item()]
	# root = edges[label_edges.index(root_idx)]
 
	"""for final_path in edges:
		path = dijkstra3d.dijkstra(1-skel, root, final_path).astype(np.int32)
		for coord_idx, coord in enumerate(path [:-1]):
			vector = coord - path[coord_idx + 1,:]
			vector_norm = vector / np.linalg.norm(vector)
			orientations[:,int(coord[0]), int(coord[1]), int(coord[2])] = vector_norm

	return orientations"""

	# 3. Final Method without ensuring root
	skeleton = skan.Skeleton(skel)
	orientations = np.zeros((3, np.shape(skel)[0], np.shape(skel)[1], np.shape(skel)[2]))
	
	## 3.1 Orientation given by abs difference among adjacent voxels (more complex task)
	"""for label in range(skeleton.n_paths):
		coords_branch_i = skeleton.path_coordinates(index = label) # (N,3)
		for coord_idx, coord in enumerate(coords_branch_i [:-1]):
			vector = abs(coord - coords_branch_i[coord_idx + 1, :])
			vector_norm = vector / np.linalg.norm(vector)
			orientations[:,int(coord[0]),int(coord[1]),int(coord[2])] = vector_norm"""
	
	## 3.2 Orientation given by difference between 1st and last voxel from nth branch
	labels = np.unique(skeleton.path_label_image())[1:] 
	for label in labels:
		coords_branch = skeleton.path_coordinates(index = label -1) # Get voxel positions that belong to a given branch
		# 1. Abs(beginning - end)
		vector = abs(coords_branch[0] - coords_branch[-1])
		vector /= np.linalg.norm(vector) # Vector computation
		for coord_branch in coords_branch: # Assign the same vector to all the branch
			orientations[:,int(coord_branch[0]),int(coord_branch[1]),int(coord_branch[2])] = np.nan_to_num(vector)

	
	return orientations


def compute_orientation_tree(volume, indices, labelled_tree, skel):
	x,y,z = np.where(volume == 1)
	
	orientation_tree = np.zeros_like(indices)
	# orientation_skel = compute_orientation_skel(skel)
	orientation_skel = compute_orientation_skel(skel, volume, labelled_tree)
	
	# Apply global orientation to whole brach
	for h,w,d in zip(x,y,z):
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
		resized_liver_portal_dmap, indices_portal, skel_portal = compute_distance_map(resized_liver_portal)

		if not os.path.isfile(os.path.join(dst_folder, info_dict['Image name'][idx].split('-')[0] + '-VE-por-labelled.nii.gz')):
			resized_liver_portal_clustered = cluster_tree(resized_liver_portal, skel_portal, indices_portal)
			output_ima = nib.Nifti1Image(resized_liver_portal_clustered, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('-')[0] + '-VE-por-labelled.nii.gz')
		else:
			resized_liver_portal_clustered = nib.load(os.path.join(dst_folder, info_dict['Image name'][idx].split('-')[0] + '-VE-por-labelled.nii.gz')).get_fdata()
		oriented_portal = compute_orientation_tree(resized_liver_portal, indices_portal, resized_liver_portal_clustered, skel_portal)

		resized_liver_hepatic = binarize(skTrans.resize(liver_hepatic, args.input_size, order = 0, preserve_range=True, anti_aliasing = True))
		resized_liver_hepatic_dmap, indices_hepatic, skel_hepatic = compute_distance_map(resized_liver_hepatic)

		if not os.path.isfile(os.path.join(dst_folder, info_dict['Image name'][idx].split('-')[0] + '-VE-hep-labelled.nii.gz')):
			resized_liver_hepatic_clustered = cluster_tree(resized_liver_hepatic, skel_hepatic, indices_hepatic)
			output_ima = nib.Nifti1Image(resized_liver_hepatic_clustered, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('-')[0] + '-VE-hep-labelled.nii.gz')
		else:
			resized_liver_hepatic_clustered = nib.load(os.path.join(dst_folder, info_dict['Image name'][idx].split('-')[0] + '-VE-hep-labelled.nii.gz')).get_fdata()
		oriented_hepatic = compute_orientation_tree(resized_liver_hepatic, indices_hepatic, resized_liver_hepatic_clustered, skel_hepatic)

		# SAVE RESIZED IMAGE
		output_ima = nib.Nifti1Image(resized_liver, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
		nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver.nii.gz')

		if args.binary:
			# SAVE RESIZED IMAGE
			
			output_portal = nib.Nifti1Image(resized_liver_portal, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_portal, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_vessel.nii.gz')
			resized_liver_portal_dmap += 1
			resized_liver_portal_dmap [resized_liver_portal == 0 ] = 0
			output_portal_dmap = nib.Nifti1Image(resized_liver_portal_dmap, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_portal_dmap, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_dmap.nii.gz')
			output_ima = nib.Nifti1Image(oriented_portal, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_ima, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_por_ori.nii.gz')		
			
			output_hepatic = nib.Nifti1Image(resized_liver_hepatic, info_dict['Affine matrix'][idx], info_dict['Header'][idx])
			nib.save(output_hepatic, dst_folder + '/' + info_dict['Image name'][idx].split('.')[0] + '-liver_hep_vessel.nii.gz')
			resized_liver_hepatic_dmap += 1
			resized_liver_hepatic_dmap [resized_liver_hepatic == 0] = 0
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
	"""for name in bar_:
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
			radius_h.append(radius)"""

	y_true_nifti = nib.load('/home/guijosa/Documents/PythonDocs/VEELA/dataset/001-VE-por.nii.gz')
	y_true  = y_true_nifti.get_fdata()

	skel = img_as_float32(skeletonize_3d(y_true)).astype(np.uint8)
	dmap, indices = DTM(1-skel, return_indices=True)

	clustered_tree = nib.load('/home/guijosa/Documents/PythonDocs/VEELA/dataset_labelled/001-VE-por-labelled.nii.gz')
	orientation_tree = compute_orientation_tree(y_true, indices)

	print('DONE')
