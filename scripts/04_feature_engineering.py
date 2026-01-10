# the goal here is to :
# extract simple image features: brightness, contrast nd saturation
# merge them with the deepface predictions from notebook 3
# at the end save a clean dataset that i will use for the ml model in notebook 5

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2


def engineer_features():
    
    BASE_PATH = Path(__file__).parent.parent
    
    train_pred_path = BASE_PATH / "results" / "baseline" / "pred_train.parquet"
    val_pred_path   = BASE_PATH / "results" / "baseline" / "pred_val.parquet"
    
    train_pred = pd.read_parquet(train_pred_path)
    val_pred   = pd.read_parquet(val_pred_path)
    
    print(train_pred.shape, val_pred.shape)
    print(train_pred.head())
    # i load the deepface prediction files from notebook 3 so i can attach my own image features on top
    
    def extract_features(img_path):
        try:
            img = cv2.imread(img_path)
    
            if img is None:
                return {"brightness": None, "contrast": None, "saturation": None}
    
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
            brightness = gray.mean()
            contrast   = gray.std()
            saturation = hsv[:, :, 1].mean()
    
            return {
                "brightness": float(brightness),
                "contrast": float(contrast),
                "saturation": float(saturation),
            }
        except:
            return {"brightness": None, "contrast": None, "saturation": None}
    # this function extracts simple image stats: i want to test if deepface fails more on dark or low contrast pictures
    
    def add_img_path(df):
        df = df.copy()
        df["img_path"] = df["file"].apply(
            lambda f: str(BASE_PATH / "data" / "processed" / "balanced_images" / 
                          ("train" if "train" in f else "val") / 
                          os.path.basename(f))
        )
        return df
    
    train_pred = add_img_path(train_pred)
    val_pred   = add_img_path(val_pred)
    
    print(train_pred.head(3))
    # rebuilt the correct path to each image so i can extract pixels in the next step
    
    def compute_features(df):
        rows = []
        for r in tqdm(df.itertuples(), total=len(df)):
            feats = extract_features(r.img_path)
            feats["file"] = r.file
            rows.append(feats)
        return pd.DataFrame(rows)
    
    train_feats = compute_features(train_pred)
    val_feats   = compute_features(val_pred)
    
    print(train_feats.head())
    # here i'm extracting brightness contrast saturation for each image
    
    out_dir = BASE_PATH / "results" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_feats_path = out_dir / "train_features.parquet"
    val_feats_path   = out_dir / "val_features.parquet"
    
    train_feats.to_parquet(train_feats_path, index=False)
    val_feats.to_parquet(val_feats_path, index=False)
    
    print(train_feats.shape, val_feats.shape)
    # saving the extracted features so i can reuse it later without recomputing
    
    train_full = train_pred.merge(train_feats, on="file", how="left")
    val_full   = val_pred.merge(val_feats, on="file", how="left")
    
    print(train_full.head())
    # by combining deepface outputs + my image features i'll have a clean dataset ready for ML part
    
    ml_dir = BASE_PATH / "data" / "ml_ready"
    ml_dir.mkdir(parents=True, exist_ok=True)
    
    train_full.to_parquet(ml_dir / "train_ml_ready.parquet", index=False)
    val_full.to_parquet(ml_dir / "val_ml_ready.parquet", index=False)
    
    print("done")
    
    #clean dataset ready for notebook 5 where i will train the model that predicts deepface errors


if __name__ == "__main__":
    engineer_features()
