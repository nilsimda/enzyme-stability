#!/bin/sh
#SBATCH --job-name=t5
#SBATCH --nodelist=gpunode10
#SBATCH --ntasks=1
#SBATCH --partition=GPU-A40
#SBATCH --gpus-per-task=1
#SBATCH --error ./t5.err
#SBATCH --output ./t5.out

srun python3 /proj/n.imdahl/embedding_scripts/prot_t5.py data/proteins/test data/t5_embeddings
