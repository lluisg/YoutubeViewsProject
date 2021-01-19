#!/bin/bash
#SBATCH -p veu
#SBATCH --mem=30G
#SBATCH --gres=gpu:2
#SBATCH --output=model_final_output.log

python tuningModel.py --name=model_final --iterations=20 --max_epochs=2000 --new_data=0
# python tuningModel.py --name=model_final --iterations=2 --max_epochs=30 --new_data=0
