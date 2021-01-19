#!/bin/bash
#SBATCH -p veu
#SBATCH --mem=30G
#SBATCH --gres=gpu:2
#SBATCH --output=model_extra_output.log

python tuningModel.py --name=model_extra --iterations=30 --max_epochs=2000 --new_data=0
