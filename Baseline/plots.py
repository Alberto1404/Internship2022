import matplotlib.pyplot as plt
import torch
import os
import random
import numpy as np

from monai.inferers import sliding_window_inference

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

def save_loss_metric(epoch_loss_values, metric_values, loss_validation, fold, args):
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
	plt.savefig(current_path + '/metric_loss_'+str(fold)+'.png')
	plt.clf()

def display_KFCV(metrics_all, args): # CADA FOLD ES EL MAXLINE MINLINE DEL EJEMPLO. TU AHORA PON EN LABELS TRAINING Y VALIDATION

	fig = plt.figure("train", (12, 6))
	x = np.arange(metrics_all.shape[-1]) + 1
	colors = ['b', 'r', 'g', 'c', 'm']
	plot_losses = []

	for fold_idx, fold in enumerate(metrics_all):
		ax1 = fig.add_subplot(1,2,1)
		ax2 = fig.add_subplot(1,2,1)
		loss_training = fold[0,:]
		loss_validation = fold[1,:]
		dice_validation = fold[2,:]

		# FIGURE 1
		fold_i_loss_tr, = ax1.plot(x,loss_training, label = 'Training', linestyle = 'dashed')
		fold_i_loss_val, = ax1.plot(x, loss_validation, label = 'Validation')
		# leg1 = ax1.legend(loc = 'upper right')
		# leg2 = ax1.legend([fold_i_loss_tr, fold_i_loss_val], ['Fold {}'.format(fold_idx+1)], loc = 'center right')
		plot_losses.append([fold_i_loss_tr, fold_i_loss_val])

		# FIGURE 2
		ax2.plot(x, dice_validation, label = 'Fold {}'.format(fold_idx+1))
		leg3 = ax2.legend()
	leg1 = ax1.legend(plot_losses[0], ['Training', 'Validation'])
	leg2 = ax1.legend([l[0] for l in plot_losses], ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5'])
	ax1.gca().add_artist(leg1)