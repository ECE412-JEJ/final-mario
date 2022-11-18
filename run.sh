#!/bin/bash
#
#SBATCH --job-name=speech_audio_synth
#SBATCH --output=/zooper2/jiyoon.pyo/proj4-mario/output.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1-1
#SBATCH --mem=20gb

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/

python talknet_training.py
