#!/bin/bash
#SBATCH --job-name=TopNet		# Choose name of your job
#SBATCH --nodes=1                   	# nombre de noeud
#SBATCH --gres=gpu:1			# Don't change this
#SBATCH --cpus-per-task=6 		# Don't change this
#SBATCH --partition=a6000		# choose GPU that you need for example: 2080GPU, a6000
#SBATCH --error=./name_output_file.txt 

singularity exec --nv /home/alberto/SIF_files/Transformers.sif python3 
/home/alberto/code/Internship2022/train.py
