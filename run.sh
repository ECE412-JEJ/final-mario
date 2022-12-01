#!/bin/bash
#
#SBATCH --job-name=speech_audio_final
#SBATCH --output=/zooper2/jiyoon.pyo/final-mario/output.out
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1-1
#SBATCH --mem=20gb

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/

cd final-mario
python phoneme_pitch_extract.py
