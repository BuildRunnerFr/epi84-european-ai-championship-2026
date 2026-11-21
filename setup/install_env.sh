#!/bin/bash
# À lancer sur le NŒUD DE LOGIN (il a internet), une seule fois pour toute l'équipe.
# Avant : chargez les modules Python / CUDA recommandés par les organisateurs.
# Usage : bash setup/install_env.sh <dossier_projet>
set -euo pipefail
DIR=${1:?"usage : bash setup/install_env.sh <dossier_projet>"}

python3 -m venv "$DIR/env"
source "$DIR/env/bin/activate"
pip install --upgrade pip

# Une seule commande pour que pip choisisse des versions compatibles entre elles :
# vLLM impose sa version de PyTorch (et de torchvision).
pip install vllm torchvision transformers accelerate bitsandbytes peft \
    scikit-learn pandas matplotlib huggingface_hub

python -c "import torch, vllm, transformers; \
print('torch', torch.__version__, '| vllm', vllm.__version__, \
'| transformers', transformers.__version__)"
echo "Environnement prêt : source $DIR/env/bin/activate"
