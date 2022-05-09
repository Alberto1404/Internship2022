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


def transformations(args):
	keys = ['image', 'vessel']
	if len(args.decoder) == 2:
		keys.append('dmap')
		keys.append('ori')
	else:
		if args.decoder[0] == 'dmap':
			keys.append('dmap')
		else:
			keys.append('ori')
	
	train_transforms = Compose(
		[
			LoadImaged(keys=keys),
			EnsureChannelFirstd(keys= keys if not 'ori' in keys else keys[:-1]),
			Orientationd(keys=keys, axcodes="RAS"),
			RandZoomd(keys=keys, min_zoom=1.1, max_zoom=2, prob = 0.3),
			RandFlipd(
				keys=keys,
				spatial_axis=[0],
				prob=0.50,
			),
			RandFlipd(
				keys=keys,
				spatial_axis=[1],
				prob=0.50,
			),
			RandFlipd(
				keys=keys,
				spatial_axis=[2],
				prob=0.50,
			),
			RandRotated(
				keys = keys,
				range_x = np.pi/9,
				range_y = np.pi/9,
				range_z = np.pi/9,
				prob = 0.50
			),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			ToTensord(keys=keys),
		]
	)
	val_transforms = Compose(
		[
			LoadImaged(keys=keys),
			EnsureChannelFirstd(keys= keys if not 'ori' in keys else keys[:-1]),
			Orientationd(keys=keys, axcodes="RAS"),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			ToTensord(keys=keys),
		]
	)

	test_transforms = Compose(
		[
			LoadImaged(keys=keys),
			EnsureChannelFirstd(keys= keys if not 'ori' in keys else keys[:-1]),
			Orientationd(keys=keys, axcodes="RAS"),
			NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
			ToTensord(keys=keys),
		]
	)
	return train_transforms, val_transforms, test_transforms


def get_loaders(args, json_routes):

	datasets = json_routes

	datalist = load_veela_datalist(datasets, "training")
	val_files = load_veela_datalist(datasets, "validation")
	test_files = load_veela_datalist(datasets, "test")
	
	train_transforms, val_transforms, test_transforms  = transformations(args)

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

