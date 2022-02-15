import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import random

from monai.inferers import sliding_window_inference


def plot_slices(model,val_ds, size, weights_dir, dictionary):
	model.load_state_dict(torch.load(os.path.join(weights_dir, "best_metric_model.pth")))
	# Visualize random slice
	slice = random.randint(1,100)
	# model.load_state_dict(torch.load('./best_metric_model.pth')) # PRETRAINED NETWORK
	model.eval()
	with torch.no_grad():
		img_name = os.path.split(val_ds[slice]["image_meta_dict"]["filename_or_obj"])[1]
		img = val_ds[slice]["image"]
		label = val_ds[slice]["label"]
		val_inputs = torch.unsqueeze(img, 1).cuda()
		val_labels = torch.unsqueeze(label, 1).cuda()
		val_outputs = sliding_window_inference(
			val_inputs, size, 4, model, overlap=0.8
			)
		plt.figure("check", (18, 6))
		plt.subplot(1, 3, 1)
		plt.title("image")
		plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, dictionary[img_name]], cmap="gray")
		plt.subplot(1, 3, 2)
		plt.title("label")
		plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, dictionary[img_name]])
		plt.subplot(1, 3, 3)
		plt.title("output")
		plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, dictionary[img_name]])
		plt.show()

def plot_loss_metric(eval_num,epoch_loss_values, metric_values):
	plt.figure("train", (12, 6))
	plt.subplot(1, 2, 1)
	plt.title("Iteration Average Loss")
	x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
	y = epoch_loss_values
	plt.xlabel("Iteration")
	plt.plot(x, y)
	plt.subplot(1, 2, 2)
	plt.title("Val Mean Dice")
	x = [eval_num * (i + 1) for i in range(len(metric_values))]
	y = metric_values
	plt.xlabel("Iteration")
	plt.plot(x, y)
	plt.show()
