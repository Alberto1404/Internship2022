#!/bin/bash
#SBATCH --job-name=TopNet		# Choose name of your job
#SBATCH --nodes=1                   	# nombre de noeud
#SBATCH --gres=gpu:1			# Don't change this
#SBATCH --cpus-per-task=6 		# Don't change this
#SBATCH --partition=a6000		# choose GPU that you need for example: 2080GPU, titanGPU
#SBATCH --error=./name_output_file.txt 

singularity exec --nv /home2/alberto/SIF_files/SIF_ALBERTO.sif python3 /home2/alberto/code_TopNet/train.py -net unetr -alpha 0 -binary True -vessel hepatic -batch 1 -epochs 2000 -lr 1e-4 -input_size 224 224 128 -metric dice -pos_embed perceptron -norm_name instance -k 1
