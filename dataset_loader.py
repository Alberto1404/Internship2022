from monai.data import(
	load_decathlon_datalist, 
	CacheDataset,
	DataLoader
)
from monai.transforms import (
	AddChanneld,
	Compose,
	CropForegroundd,
	LoadImaged,
	Orientationd,
	RandFlipd,
	RandCropByPosNegLabeld,
	RandShiftIntensityd,
	NormalizeIntensityd,
	Spacingd,
	RandRotate90d,
	ToTensord,
)
import os
from utils import load_veela_datalist

import json
def transformations(size):
	train_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			AddChanneld(keys=["image", "label"]),
			Spacingd(
				keys=["image", "label"],
				pixdim=(1.0, 1.0, 1.0),
				mode=("bilinear", "nearest"),
			),
			Orientationd(keys=["image", "label"], axcodes="RAS"),
			# ScaleIntensityRanged(
			#     keys=["image"],
			#     a_min=-175,
			#     a_max=250,
			#     b_min=0.0,
			#     b_max=1.0,
			#     clip=True,
			# ),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			CropForegroundd(keys=["image", "label"], source_key="image"),
			RandCropByPosNegLabeld(
				keys=["image", "label"],
				label_key="label",
				spatial_size=size,
				pos=1,
				neg=1,
				num_samples=4,
				image_key="image",
				image_threshold=0,
			),
			RandFlipd(
				keys=["image", "label"],
				spatial_axis=[0],
				prob=0.10,
			),
			RandFlipd(
				keys=["image", "label"],
				spatial_axis=[1],
				prob=0.10,
			),
			RandFlipd(
				keys=["image", "label"],
				spatial_axis=[2],
				prob=0.10,
			),
			RandRotate90d(
				keys=["image", "label"],
				prob=0.10,
				max_k=3,
			),
			RandShiftIntensityd(
				keys=["image"],
				offsets=0.10,
				prob=0.50,
			),
			ToTensord(keys=["image", "label"]),
		]
	)
	val_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			AddChanneld(keys=["image", "label"]),
			Spacingd(
				keys=["image", "label"],
				pixdim=(1.0, 1.0, 1.0),
				mode=("bilinear", "nearest"),
			),
			Orientationd(keys=["image", "label"], axcodes="RAS"),
			# ScaleIntensityRanged(
			#     keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
			# ),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			CropForegroundd(keys=["image", "label"], source_key="image"),
			ToTensord(keys=["image", "label"]),
		]
	)

	test_transforms = Compose(
		[
			LoadImaged(keys=["image", "label"]),
			AddChanneld(keys=["image", "label"]),
			Spacingd(
				keys=["image", "label"],
				pixdim=(1.0, 1.0, 1.0),
				mode=("bilinear", "nearest"),
			),
			Orientationd(keys=["image", "label"], axcodes="RAS"),
			# ScaleIntensityRanged(
			#     keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
			# ),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			CropForegroundd(keys=["image", "label"], source_key="image"),
			ToTensord(keys=["image", "label"]),
		]
	)
	return train_transforms, val_transforms, test_transforms


def get_train_valid_loader(size, json_routes):

	datasets = json_routes[0] # BAD!!!!!!!! K-FOLD CROSS VALIDATION !!!!!

	datalist = load_veela_datalist(datasets, "training")
	val_files = load_veela_datalist(datasets, "validation")
	test_files = load_veela_datalist(datasets, "test")
	
	train_transforms, val_transforms, test_transforms  = transformations(size)

	train_ds = CacheDataset(
		data=datalist,
		transform=train_transforms,
		cache_num=23,
		cache_rate=1.0,
		num_workers=8,
	)
	train_loader = DataLoader(
		train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
	)
	val_ds = CacheDataset(
		data=val_files, transform=val_transforms, cache_num=5, cache_rate=1.0, num_workers=4
	)
	val_loader = DataLoader(
		val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
	)
	test_ds = CacheDataset(
		data = test_files, transform=test_transforms, cache_num = 7, cache_rate =1,	num_workers=4
	)
	test_loader = DataLoader(
		test_ds, batch_size=1, shuffle = False, num_workers=2, pin_memory=True
	)
	return train_loader, val_loader, test_loader, val_ds

