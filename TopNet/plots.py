import matplotlib.pyplot as plt
import torch
import os
import random
import numpy as np

from monai.inferers import sliding_window_inference
from scipy.signal import unit_impulse

current_path = os.path.abspath(os.getcwd())

def save_loss_metric(loss_training, metric_values, loss_validation, fold, best_epoch, args):

	# PLOT TRAINING VS VALIDATION LOSS & METRIC
	x = np.arange(len(loss_training)) + 1

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
	plt.clf()
	
	# COMPARISON LOSSES AMONG DECODERS
	"""x = np.arange(len(metric_values)) + 1

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
	fig.clf()"""
