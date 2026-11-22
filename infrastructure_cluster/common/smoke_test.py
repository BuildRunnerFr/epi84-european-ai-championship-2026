"""Test de fumée : à lancer dans un job GPU (sbatch slurm/test.sh).

Vérifie les GPU, le bf16, la vitesse d'un produit matriciel, les imports des
bibliothèques et, si MODEL_PATH est défini, la présence du tokenizer du modèle
téléchargé (sans internet).
"""
import os
import sys
import time

import torch


def main():
    print("Python", sys.version.split()[0], "| PyTorch", torch.__version__)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not torch.cuda.is_available():
        print("ÉCHEC : aucun GPU visible")
        sys.exit(1)
    n = torch.cuda.device_count()
    print(f"{n} GPU visible(s)")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i} : {p.name}, {p.total_memory / 1e9:.1f} Go")
    print("bf16 supporté :", torch.cuda.is_bf16_supported())

    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    x @ x  # chauffe
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        x @ x
    torch.cuda.synchronize()
    tflops = 10 * 2 * 8192 ** 3 / (time.time() - t) / 1e12
    print(f"Produit matriciel bf16 sur le GPU 0 : {tflops:.0f} TFLOPS")

    for name in ["transformers", "vllm", "sklearn", "bitsandbytes", "peft",
                 "torchvision"]:
        try:
            __import__(name)
            print(f"  import {name} : ok")
        except Exception as e:  # on veut la liste complète des problèmes
            print(f"  import {name} : ÉCHEC ({e.__class__.__name__}: {e})")

    print("HF_HOME =", os.environ.get("HF_HOME"),
          "| HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"))
    model_path = os.environ.get("MODEL_PATH")
    if model_path and not os.path.isdir(model_path):
        print(f"MODEL_PATH ignoré : dossier introuvable ({model_path})")
    elif model_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        print(f"Tokenizer chargé hors ligne depuis {model_path} "
              f"({len(tok)} tokens dans le vocabulaire)")
    print("TEST DE FUMÉE TERMINÉ")


if __name__ == "__main__":
    main()
