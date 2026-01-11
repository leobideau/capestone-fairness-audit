# goal of this notebook:
# this notebook extracts visual face embeddings from all images using deepface.represent()
# these embeddings will later be merged with my existing ml features to improve the models as my other features didnt help much
# Nb: i do NOT touch any of my previous parquet files 
# i simply create new embedding files and keep everything modular and safe 

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from deepface import DeepFace


def extract_embeddings():
    
    BASE_PATH = Path(__file__).parent.parent
    
    train_parquet = BASE_PATH / "data" / "ml_ready" / "train_ml_ready.parquet"
    val_parquet   = BASE_PATH / "data" / "ml_ready" / "val_ml_ready.parquet"
    
    out_dir = BASE_PATH / "data" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_emb_out = out_dir / "train_embeddings.parquet"
    val_emb_out = out_dir / "val_embeddings.parquet"
    
    # Skip if embeddings already exist
    if train_emb_out.exists() and val_emb_out.exists():
        print(f"Embeddings already exist, skipping extraction")
        print(f"   - {train_emb_out}")
        print(f"   - {val_emb_out}")
        return
    
    print("train parquet:", train_parquet)
    print("val parquet:", val_parquet)
    print("output dir:", out_dir)
    # loading the 'ml_ready' datasets i built earlier and preparing a new folder where i will store clean embedding parquet files
    
    train_df = pd.read_parquet(train_parquet)
    val_df   = pd.read_parquet(val_parquet)
    
    print(train_df.shape, val_df.shape)
    print(train_df.head())
    # reading the ml-ready train/val datasets that contain all metadata, deepface predictions and image paths
    # i will only use the img_path column here to extract embeddings
    
    from multiprocessing import Pool, TimeoutError as MPTimeoutError
    from functools import partial
    
    def get_embedding_safe(path):
        """Wrapper for multiprocessing"""
        try:
            res = DeepFace.represent(
                img_path = path,
                model_name = "Facenet512",
                enforce_detection = False
            )
            return res[0]["embedding"]
        except Exception as e:
            return None
    
    def get_embedding(path):
        """Get embedding with timeout using multiprocessing"""
        with Pool(1) as pool:
            try:
                result = pool.apply_async(get_embedding_safe, (path,))
                embedding = result.get(timeout=5)  # 5 second timeout
                return embedding
            except MPTimeoutError:
                print(f"\nTimeout on {path.split('/')[-1]}, skipping...")
                pool.terminate()
                pool.join()
                return None
            except Exception as e:
                return None
    # this is the core function that extracts a 512 dimensional embedding vector from each image
    # it's stable, and it will not crash the notebook because i catch every exception and return none if needed
    # uses multiprocessing with timeout to skip images that freeze deepface (works on Mac)
    
    train_df["embedding"] = [
        get_embedding(p) for p in tqdm(train_df["img_path"], desc="train embeddings")
    ]
    
    train_df.to_parquet(train_emb_out, index=False)
    
    print("saved:", train_emb_out)
    # running face embedding extraction on the train set
    # each embedding is stored inside the dataframe, then exported to a dedicated parquet file.
    
    val_df["embedding"] = [
        get_embedding(p) for p in tqdm(val_df["img_path"], desc="val embeddings")
    ]
    
    val_df.to_parquet(val_emb_out, index=False)
    
    print("saved:", val_emb_out)
    # same for the validation set
    # i have two brand-new files without touching previous work: train_embeddings.parquet nd val_embeddings.parquet
    
    print(train_df["embedding"].head())
    # confirming that embeddings were correctly added
    
    # next step will happen in notebook 4BIS : merging embeddings + existing ML features


if __name__ == "__main__":
    extract_embeddings()
