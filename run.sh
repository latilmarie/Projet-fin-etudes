#!/bin/bash
#SBATCH --gres=gpu:1 --constraint=gpu1080
#SBATCH -p grantgpu -A g2024a304g
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --tasks-per-node 1
#--mail-type=ALL
#--mail-user=mlatil@unistra.fr

module load gdal/gdal-3.7.2.gcc11
activate deep-learning

output_filename=$(python -c 'from params import output_filename; print(output_filename)')
python -u main.py > "$output_filename"