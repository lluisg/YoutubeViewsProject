#!/bin/bash
#SBATCH -p veu
#SBATCH --mem=30G
#SBATCH --gres=gpu:2
#SBATCH --output=output_model22.log

python tuningModel.py --name=model22 --mini=1 --new=0 --save_training=1 --save_outputs=1
