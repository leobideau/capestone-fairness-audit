# goal: run an existing model (deepface) on my balanced database to do predictions    

import os, pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm
from deepface import DeepFace


def run_deepface_predictions():
    
    BASE_PATH = Path(__file__).parent.parent
    
    DATA_DIR = BASE_PATH / "data" / "processed" / "balanced_images"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TRAIN_CSV = BASE_PATH / "data" / "processed" / "balanced_train.csv"
    VAL_CSV = BASE_PATH / "data" / "processed" / "balanced_val.csv"
    OUT_DIR = BASE_PATH / "results" / "baseline"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(TRAIN_DIR)
    print(TRAIN_CSV)
    print(OUT_DIR)
    # setting all the paths nedded
    
    def attach_path(df, img_dir):
        df = df.copy()
        df["fname"] = df["file"].apply(lambda f: os.path.basename(f))
        df["img_path"] = df["fname"].apply(lambda f: os.path.join(img_dir, f))
        return df
    
    train_df = attach_path(pd.read_csv(TRAIN_CSV), str(TRAIN_DIR))
    val_df = attach_path(pd.read_csv(VAL_CSV), str(VAL_DIR))
    print(len(train_df), len(val_df), train_df.head(2))
    # linking each file in the csv with its own actual image path
    
    FAIR2COARSE = {
        "White":"white", "Black":"black", "Indian":"indian",
        "Middle Eastern":"middle eastern", "Latino_Hispanic":"latino hispanic",
        "East Asian":"asian", "Southeast Asian":"asian",
    }
    GENDER_MAP_PRED2GT = {"Man":"Male", "Woman":"Female"}
    # here i'm alligning deepface outputs with fairface label format
    
    def analyze_one(img_path, detector="retinaface"):
        try:
            res = DeepFace.analyze(
                img_path,                              
                actions=["gender", "race"],
                detector_backend=detector            
            )

            if isinstance(res, list):
                res = res[0]

            return {
                "pred_gender":      res.get("dominant_gender"),
                "pred_gender_score":res.get("gender", {}).get(res.get("dominant_gender")),
                "pred_race":        res.get("dominant_race"),
                "pred_race_score":  res.get("race", {}).get(res.get("dominant_race")),
                "error":            None,
            }

        except Exception as e:
            return {
                "pred_gender": None, "pred_gender_score": None,
                "pred_race":   None, "pred_race_score":   None,
                "error": str(e),
            }
    # I just use deepface on one image and pull out the gender nd race outputs i need,
    # i'm keeping the call minimal because deepface changes a lot between versions.
    # If it returns multiple faces, I keep the first. If it errors, I store the error
    
    def run_split(df, out_path, detector="retinaface", save_every=50):
        rows = []
        out_df = None
        
        if os.path.exists(out_path):
            try:
                out_df = pd.read_parquet(out_path)
                done_files = set(out_df['file'].tolist())
            except:
                out_df = None
                done_files = set()
        else:
            done_files = set()
        
        for i, r in enumerate(df.itertuples(), start=1):
            if r.file in done_files:
                continue
            
            pred = analyze_one(r.img_path, detector)
            pred.update({
                "file": r.file,
                "race_true": r.race,
                "gender_true": r.gender,
            })
            rows.append(pred)
            
            if i % save_every == 0:
                tmp_df = pd.DataFrame(rows)
                if out_df is not None:
                    out_df = pd.concat([out_df, tmp_df], ignore_index=True)
                else:
                    out_df = tmp_df
                out_df.to_parquet(out_path, index=False)
                rows = []
        
        if rows:
            tmp_df = pd.DataFrame(rows)
            if out_df is not None:
                out_df = pd.concat([out_df, tmp_df], ignore_index=True)
            else:
                out_df = tmp_df
            out_df.to_parquet(out_path, index=False)
        
        return out_df
    # I use this function to run DeepFace on the split train or val
    # it also supports restarting if the process crashes:
    # if a partial parquet file already exists, it will be reloaded and only goes trough the remaining images
    # Nb : it saves progress every 50 images so i don't lose everything if the kernel crashes (which happened before)
    
    pred_train = run_split(train_df, str(OUT_DIR / "pred_train.parquet"))
    pred_val = run_split(val_df, str(OUT_DIR / "pred_val.parquet"))
    print(pred_train.head(3))
    # run deepface on the full train nd val splits and save the results
    # Nb: this will take a while because it goes through the 10'000 images one by one
    
    p = str(OUT_DIR / "pred_train.parquet")
    print("exists:", os.path.exists(p))
    
    if os.path.exists(p):
        try:
            df = pd.read_parquet(p)
            print("rows:", len(df))
            print(df.head())
        except:
            print("the file is corrupted or empty")
    # quick sanity check to see if the train predictions file was created correctly
    
    print("pred_train exists:", os.path.exists(str(OUT_DIR / "pred_train.parquet")))
    print("pred_val exists:", os.path.exists(str(OUT_DIR / "pred_val.parquet")))
    # check both prediction worked
    
    train_pred = pd.read_parquet(str(OUT_DIR / "pred_train.parquet"))
    val_pred = pd.read_parquet(str(OUT_DIR / "pred_val.parquet"))
    
    print(len(train_pred), len(val_pred))
    print(train_pred.head())
    # loaded the prediction files for train and val to check that everything is there and that the shapes are ok


if __name__ == "__main__":
    run_deepface_predictions()
