import matplotlib.pyplot as plt
import torch
import os
import random
import numpy as np

from monai.inferers import sliding_window_inference
from scipy.signal import unit_impulse

current_path = os.path.abspath(os.getcwd())


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

"""def save_loss_metric(epoch_loss_values, metric_values, loss_validation, args):
	plt.figure("train", (12, 6))
	plt.subplot(1, 2, 1)
	if args.metric == 'softdice':
		plt.title("ClDice Loss")
	else:
		plt.title("Dice Cross Entropy Loss")
	x = np.arange(len(epoch_loss_values)) + 1
	y = epoch_loss_values
	y2 = loss_validation
	plt.xlabel("Epoch")
	plt.plot(x, y)
	plt.plot(x,y2)
	plt.legend(['Training', 'Validation'])
	plt.subplot(1, 2, 2)
	if args.metric == 'dice':
		plt.title("Validation dice")
	elif args.metric == 'haus':
		plt.title('Validation Haussdorf distance')
	elif args.metric == 'surfdist':
		plt.title('Validation Surface distance')
	else:
		plt.title('Validation Soft Cldice metric')
	x = np.arange(len(metric_values)) + 1
	y = metric_values
	plt.xlabel("Epoch")
	plt.plot(x, y)
	plt.savefig(current_path + '/metric_loss.png')"""


def save_loss_metric(loss_training, metric_values, loss_validation, fold, best_epoch, args):

	# PLOT HABITUAL DE TRAINING Y VALIDATION
	"""x = np.arange(len(loss_training)) + 1

	plt.figure('training', (12,6))

	plt.subplot(1,2,1)
	plt.title('Loss')
	y = loss_training
	y2 = loss_validation
	plt.xlabel("Epoch")
	plt.plot(x, y)
	plt.plot(x,y2)
	unit = unit_impulse(len(x), best_epoch - 1)
	unit[unit != 1] = -1
	plt.plot(x, (unit * y2), 'g*')
	plt.ylim([0, 2.6])
	plt.legend(['Training', 'Validation', 'Saved'])

	plt.subplot(1,2,2)
	plt.title('Metric')
	y = metric_values
	plt.plot(x, y)
	plt.plot(x,unit * y, 'g*')
	plt.ylim([0, 1])
	plt.legend(['Validation dice', 'Saved'])
	plt.savefig(current_path + '/metric_loss_'+str(fold)+'.png')
	plt.clf()"""

	x = np.arange(len(metric_values)) + 1

	fig = plt.figure("train", (12, 6))

	ax0 = fig.add_subplot(1,2,1)
	ax1 = ax0.twinx()
	ax2 = fig.add_subplot(1,2,2)
	# ax3 = ax2.twinx()

	# ax1.get_shared_y_axes().join(ax1, ax3)
	c1, = ax0.plot(x, loss_training[1][-args.epochs:], c = 'tab:blue')
	c2, = ax0.plot(x, loss_validation[1][-args.epochs:], c = 'tab:green')
	c3, = ax1.plot(x, loss_training[2][-args.epochs:], c = 'tab:orange')
	c4, = ax1.plot(x, loss_validation[2][-args.epochs:], 'tab:pink')

	if args.decoder[0] == 'dmap':
		ax0.legend([c1,c2,c3,c4], ['Train vessel loss', 'Val vessel loss', 'Train dmap loss', 'Val dmap loss'])
	if args.decoder[0] == 'ori':
		ax0.legend([c1,c2,c3,c4], ['Train vessel loss', 'Val vessel loss', 'Train ori loss', 'Val ori loss'])
	c5, = ax2.plot(x, metric_values)
	ax2.legend([c5],['Validation dice'])
	plt.savefig(current_path + '/metric_loss_'+str(fold)+'.png')
	fig.clf()
