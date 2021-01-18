#!/bin/bash
#SBATCH -p veu
#SBATCH --mem=30G
#SBATCH --gres=gpu:2
#SBATCH --output=model1_output.log

python tuningModel.py --name=model1 --input_elements=1245 --mini=1 --new_model=1 \
    --new_data=0 --save_training=1 --save_outputs=1 --iterations=2 --max_epochs=20
