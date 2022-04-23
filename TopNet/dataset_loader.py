import numpy as np

from monai.data import(
	load_decathlon_datalist, 
	CacheDataset,
	DataLoader
)
from monai.transforms import (
	EnsureChannelFirstd,
	Compose,
	LoadImaged,
	RandZoomd,
	Orientationd,
	RandFlipd,
	RandRotated,
	NormalizeIntensityd,
	ToTensord,
)
from utils import load_veela_datalist


def transformations(size):
	train_transforms = Compose(
		[
			LoadImaged(keys=["image", "vessel", "dmap"]),
			EnsureChannelFirstd(keys=["image", "vessel", "dmap"]),
			Orientationd(keys=["image", "vessel", "dmap"], axcodes="RAS"),
			RandZoomd(keys=["image", "vessel", "dmap"], min_zoom=1.1, max_zoom=2, prob = 0.3),
			RandFlipd(
				keys=["image", "vessel", "dmap"],
				spatial_axis=[0],
				prob=0.50,
			),
			RandFlipd(
				keys=["image", "vessel", "dmap"],
				spatial_axis=[1],
				prob=0.50,
			),
			RandFlipd(
				keys=["image", "vessel", "dmap"],
				spatial_axis=[2],
				prob=0.50,
			),
			RandRotated(
				keys = ["image", "vessel", "dmap"],
				range_x = np.pi/9,
				range_y = np.pi/9,
				range_z = np.pi/9,
				prob = 0.50
			),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			ToTensord(keys=["image", "vessel", "dmap"]),
		]
	)
	val_transforms = Compose(
		[
			LoadImaged(keys=["image", "vessel", "dmap"]),
			EnsureChannelFirstd(keys=["image", "vessel", "dmap"]),
			Orientationd(keys=["image", "vessel", "dmap"], axcodes="RAS"),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			ToTensord(keys=["image", "vessel", "dmap"]),
		]
	)

	test_transforms = Compose(
		[
			LoadImaged(keys=["image", "vessel", "dmap"]),
			EnsureChannelFirstd(keys=["image", "vessel", "dmap"]),
			Orientationd(keys=["image", "vessel", "dmap"], axcodes="RAS"),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			ToTensord(keys=["image", "vessel", "dmap"]),
		]
	)
	return train_transforms, val_transforms, test_transforms


def get_loaders(args, json_routes):

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
	return train_loader, val_loader, test_loader

