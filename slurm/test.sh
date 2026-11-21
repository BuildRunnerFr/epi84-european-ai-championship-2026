#!/bin/bash
# Test de fumée sur un nœud GPU (10 minutes). À lancer depuis la racine du dépôt :
#   sbatch slurm/test.sh
# Remplissez d'abord slurm/env.sh et les valeurs entre chevrons ci-dessous. La partition
# develbooster est celle de test sur JUWELS Booster : adaptez-la aux consignes.
#SBATCH --account=<projet>
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=logs/test-%j.out
##SBATCH --reservation=<nom>        # si les organisateurs en fournissent une

source slurm/env.sh || exit 1
nvidia-smi
nvidia-smi topo -m                    # NV# = GPU reliés par NVLink
python common/smoke_test.py
