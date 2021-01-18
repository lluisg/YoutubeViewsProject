#!/bin/bash
#SBATCH -p veu
#SBATCH --mem=30G
#SBATCH --gres=gpu:2
#SBATCH --output=model_final_output.log

python tuningModel.py --name=model_final --iterations=20
