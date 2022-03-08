import numpy as np

from monai.data import(
	load_decathlon_datalist, 
	CacheDataset,
	DataLoader
)
from monai.transforms import (
	AddChanneld,
	EnsureChannelFirstd,
	Compose,
	LoadImaged,
	MapTransform,
	Orientationd,
	RandFlipd,
	RandRotated,
	RandCropByPosNegLabeld,
	RandShiftIntensityd,
	NormalizeIntensityd,
	RandRotate90d,
	ToTensord,
)
from utils import load_veela_datalist


class Convert2MultiChanneld(MapTransform):
	"""
	Convert labels to multi channels based on brats classes:
	label 1 is the peritumoral edema
	label 2 is the GD-enhancing tumor
	label 3 is the necrotic and non-enhancing tumor core
	The possible classes are TC (Tumor core), WT (Whole tumor)
	and ET (Enhancing tumor).

	"""
	def __call__(self, data):
		d = dict(data)
		for key in self.keys:
			result = []
			# merge label 2 and label 3 to construct TC
			result.append(np.logical_or(d[key] == 2, d[key] == 3))
			# merge labels 1, 2 and 3 to construct WT
			result.append(
				np.logical_or(
					np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
				)
			)
			# label 2 is ET
			result.append(d[key] == 2)
			d[key] = np.stack(result, axis=0).astype(np.float32)
		return d


def transformations(size):
	train_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			# AddChanneld(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image", "label"]),
			# Convert2MultiChanneld(keys="label"),
			Orientationd(keys=["image", "label"], axcodes="RAS"),
			# RandCropByPosNegLabeld(
			# 	keys=["image", "label"],
			# 	label_key="label",
			# 	spatial_size=size,
			# 	pos=1,
			# 	neg=1,
			# 	num_samples=4,
			# 	image_key="image",
			# 	image_threshold=0,
			# ),
			RandFlipd(
				keys=["image", "label"],
				spatial_axis=[0],
				prob=0.50,
			),
			RandFlipd(
				keys=["image", "label"],
				spatial_axis=[1],
				prob=0.50,
			),
			RandFlipd(
				keys=["image", "label"],
				spatial_axis=[2],
				prob=0.50,
			),
			RandRotated(
				keys = ["image", "label"],
				range_x = np.pi/9,
				range_y = np.pi/9,
				range_z = np.pi/9,
				prob = 0.50
			),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			ToTensord(keys=["image", "label"]),
		]
	)
	val_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			# AddChanneld(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image", "label"]),
			# Convert2MultiChanneld(keys="label"),
			Orientationd(keys=["image", "label"], axcodes="RAS"),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			ToTensord(keys=["image", "label"]),
		]
	)

	test_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			# AddChanneld(keys=["image", "label"]),
			EnsureChannelFirstd(keys=["image", "label"]),
			# Convert2MultiChanneld(keys="label"),
			Orientationd(keys=["image", "label"], axcodes="RAS"),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			ToTensord(keys=["image", "label"]),
		]
	)
	return train_transforms, val_transforms, test_transforms


def get_train_valid_loader(args, json_routes):

	# datasets = json_routes[0]
	datasets = json_routes

	datalist = load_veela_datalist(datasets, "training")
	val_files = load_veela_datalist(datasets, "validation")
	test_files = load_veela_datalist(datasets, "test")
	
	train_transforms, val_transforms, test_transforms  = transformations(args.input_size)

	train_ds = CacheDataset(
		data=datalist,
		transform=train_transforms,
		# cache_num=23,
		cache_rate=1.0,
		num_workers=4,
	)
	train_loader = DataLoader(
		train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True
	)
	val_ds = CacheDataset(
		data=val_files, 
		transform=val_transforms, 
		# cache_num=5, 
		cache_rate=1.0,
		num_workers=4
	)
	val_loader = DataLoader(
		val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True
	)
	test_ds = CacheDataset(
		data = test_files, 
		transform=test_transforms, 
		# cache_num = 7,
		cache_rate=1.0, 
		num_workers=4
	)
	test_loader = DataLoader(
		test_ds, batch_size=args.batch, shuffle = False, num_workers=4, pin_memory=True
	)
	return train_loader, val_loader, test_loader, val_ds

