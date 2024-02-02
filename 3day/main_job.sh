#!/bin/bash
#SBATCH --job-name=bkft
#SBATCH --output=bkft-%j.out
#SBATCH --error=bkft-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smoeller@ufl.edu
#SBATCH --mem=7gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=6:00:00

date;hostname;pwd

module purge

module load fairseq/0.12.2
python /home/smoeller/fq_transformer_wu.py /blue/smoeller/smoeller/AIwkshp/

date