import os
import numpy as np
import subprocess
import sys
from main import main
import rarfile

def test_func():
    command = [
        "python", "main.py",
        "--modality", "video",
        "--extract-feats",
        "--config-path", "configs/lrw_resnet18_dctcn_boundary.json",  
        "--model-path", "models/lrw_resnet18_dctcn_video_boundary.pth",
        "--mouth-patch-path", "datasets/visual_data/00000002026.npz",
        "--mouth-embedding-out-path", "results/embeddings/test_embeddings.npz"
    ]
    
    
    
    
    print("Запуск команды:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
   
    return result.returncode

if __name__ == "__main__":
    path_f = "datasets/visual_data/00000002026.npz"
    if not os.path.isfile(path_f):
        with rarfile.RarFile("data.rar", "r") as f:
            f.extractall("datasets/visual_data")
        
    test_func()