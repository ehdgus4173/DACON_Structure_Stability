import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import TEST_DIR, SAMPLE_SUBMISSION_CSV, CHECKPOINT_DIR, PROJECT_ROOT
from dataset import MultiViewDataset, get_transforms
from models import MultiViewResNet

PHYS_COLS_V2 = [
    't_compactness', 'f_cx_offset', 't_left_mass_ratio',
    't_cx_offset', 'f_mass_upper_ratio',
    'FS_overturning', 'kern_ratio', 'effective_eccentricity',
    'eccentric_combined', 'p_delta_eccentricity'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', type=str, default='submission.csv')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test sample submission file
    test_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    
    # [NEW] Load physical features
    feature_csv_path = PROJECT_ROOT / "features" / "combined_features_v3.csv"
    feature_df = pd.read_csv(feature_csv_path)
    
    _, test_transform = get_transforms()
    test_dataset = MultiViewDataset(
        df=test_df, root_dir=str(TEST_DIR), transform=test_transform, 
        is_test=True, feature_df=feature_df, feature_cols=PHYS_COLS_V2
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load Model (Match feature branch dimension)
    model = MultiViewResNet(num_classes=1, num_phys_features=len(PHYS_COLS_V2)).to(device)
    best_model_path = CHECKPOINT_DIR / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded checkpoint from: {best_model_path}")
    else:
        print(f"Warning: Checkpoint not found at {best_model_path}! Using untrained model.")
        
    model.eval()
    all_probs = []

    print("Running inference with Physical Fusion on test set...")
    with torch.no_grad():
        for views, feats in tqdm(test_loader, desc="Inference"):
            views = [v.to(device) for v in views]
            feats = feats.to(device)
            
            outputs = model(views, feats).view(-1)
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
