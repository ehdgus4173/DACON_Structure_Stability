import pandas as pd
import numpy as np

# dev.csv 라벨 로드
dev_df = pd.read_csv('../structure-stability/data/dev.csv')
print(f"Dev 샘플 수: {len(dev_df)}")
print(f"컬럼: {list(dev_df.columns)}")
print()
print("=== 라벨 분포 ===")
print(dev_df['label'].value_counts())
print(f"stable 비율: {(dev_df['label']=='stable').sum()/len(dev_df)*100:.1f}%")
print(f"unstable 비율: {(dev_df['label']=='unstable').sum()/len(dev_df)*100:.1f}%")
