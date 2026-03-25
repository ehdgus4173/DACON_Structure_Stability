import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import TEST_DIR, SAMPLE_SUBMISSION_CSV, CHECKPOINT_DIR
from dataset import MultiViewDataset, get_transforms
from models import MultiViewResNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', type=str, default='submission.csv')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test sample submission file
    test_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    
    _, test_transform = get_transforms()
    test_dataset = MultiViewDataset(test_df, str(TEST_DIR), test_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load Model
    model = MultiViewResNet().to(device)
    best_model_path = CHECKPOINT_DIR / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded checkpoint from: {best_model_path}")
    else:
        print(f"Warning: Checkpoint not found at {best_model_path}! Using untrained model.")
        
    model.eval()
    all_probs = []

    print("Running inference on test set...")
    with torch.no_grad():
        for views in tqdm(test_loader, desc="Inference"):
            views = [v.to(device) for v in views]
            
            outputs = model(views).view(-1)
            probs = torch.sigmoid(outputs)
            
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    
    # Create final submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'unstable_prob': all_probs,
        'stable_prob': 1.0 - all_probs
    })
    
    submission.to_csv(args.output, encoding='UTF-8-sig', index=False)
    print(f"Saved predictions to {args.output}")

if __name__ == '__main__':
    main()
