#!/bin/bash

#SBATCH --job-name=service-discovery
#SBATCH --account=Project_2007433
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN #uncomment to enable mail
#SBATCH --mail-type=FAIL #uncomment to enable mail
#SBATCH --mail-type=END #uncomment to enable mail
#SBATCH --gres=gpu:v100:3

module load pytorch

python main.py > output.txt
#python test.py > output1.txt
