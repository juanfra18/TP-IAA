import torch
import os
import shutil

def save_best_model():
    print("Consolidating 5-fold ensemble into 'best_model.pth'...")
    
    ensemble_state = {}
    
    for i in range(1, 6):
        model_path = f"results/improved_model_fold{i}.pth"
        if os.path.exists(model_path):
            print(f"Loading {model_path}...")
            state_dict = torch.load(model_path)
            ensemble_state[f'fold_{i}'] = state_dict
        else:
            print(f"Error: {model_path} not found!")
            return

    output_path = "best_model.pth"
    torch.save(ensemble_state, output_path)
    print(f"✅ Successfully saved ensemble model to {output_path}")
    
    print("Verifying load...")
    loaded = torch.load(output_path)
    print(f"Loaded keys: {list(loaded.keys())}")

if __name__ == "__main__":
    save_best_model()
