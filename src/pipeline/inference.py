import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import TEST_DIR, SAMPLE_SUBMISSION_CSV, CHECKPOINT_DIR, PROJECT_ROOT, PHYS_COLS_V2
from src.dataset import MultiViewDataset, get_transforms
from src.models import MultiViewResNet


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output',     type=str, default='submission.csv')
    args = parser.parse_args(args if args is not None else [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_df    = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    feature_df = pd.read_csv(PROJECT_ROOT / "features" / "combined_features_v3.csv")

    _, test_transform = get_transforms()
    test_dataset = MultiViewDataset(
        df=test_df, root_dir=str(TEST_DIR), transform=test_transform,
        is_test=True, feature_df=feature_df, feature_cols=PHYS_COLS_V2
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultiViewResNet(num_classes=1, num_phys_features=len(PHYS_COLS_V2)).to(device)
    best_model_path = CHECKPOINT_DIR / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded checkpoint: {best_model_path}")
    else:
        print(f"⚠️  Checkpoint not found: {best_model_path} — using untrained model")

    model.eval()
    all_probs = []

    with torch.no_grad():
        for views, feats in tqdm(test_loader, desc="Inference"):
            views = [v.to(device) for v in views]
            feats = feats.to(device)
            probs = torch.sigmoid(model(views, feats).view(-1))
            all_probs.extend(probs.cpu().numpy())

    all_probs  = np.array(all_probs)
    submission = pd.DataFrame({
        'id':           test_df['id'],
        'unstable_prob': all_probs,
        'stable_prob':   1.0 - all_probs,
    })
    submission.to_csv(args.output, encoding='UTF-8-sig', index=False)
    print(f"✅ Saved: {args.output}")

if __name__ == '__main__':
    main()
