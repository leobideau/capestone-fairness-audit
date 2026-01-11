# goal of this notebook:
# merge the visual embeddings i extracted in notebook 2bis with the ML ready features from notebook 4
# Nb: no old parquet files are modified
# i'll create new datasets train_ml_final nd val_ml_final including:
# deepface score features, brightness, contrast, saturation nd the face embeddings
# these final datasets will be used in notebook 5BIS to train stronger ml models

import os
import pandas as pd
import numpy as np
from pathlib import Path


def merge_features_with_embeddings():
    
    BASE_PATH = Path(__file__).parent.parent
    
    train_ml_path = BASE_PATH / "data" / "ml_ready" / "train_ml_ready.parquet"
    val_ml_path   = BASE_PATH / "data" / "ml_ready" / "val_ml_ready.parquet"
    
    train_emb_path = BASE_PATH / "data" / "embeddings" / "train_embeddings.parquet"
    val_emb_path   = BASE_PATH / "data" / "embeddings" / "val_embeddings.parquet"
    
    out_dir = BASE_PATH / "data" / "ml_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(train_ml_path)
    print(val_ml_path)
    print(train_emb_path)
    print(val_emb_path)
    print(out_dir)
    # defining all source parquet files and creating the output directory for the final merged datasets
    
    train_ml = pd.read_parquet(train_ml_path)
    val_ml   = pd.read_parquet(val_ml_path)
    
    train_emb = pd.read_parquet(train_emb_path)
    val_emb   = pd.read_parquet(val_emb_path)
    
    print(train_ml.shape, train_emb.shape)
    print(val_ml.shape, val_emb.shape)
    # loading both ml ready nd embedding enhanced datasets
    
    def expand_embeddings(df):
        emb_matrix = np.vstack(df["embedding"].values)
        emb_cols = [f"emb_{i}" for i in range(emb_matrix.shape[1])]
        emb_df = pd.DataFrame(emb_matrix, columns=emb_cols)
        return emb_df
    
    train_emb_expanded = expand_embeddings(train_emb)
    val_emb_expanded   = expand_embeddings(val_emb)
    
    print(train_emb_expanded.shape)
    print(val_emb_expanded.shape)
    # converting the embedding lists into a clean 512 column dataframe where each row becomes a 512 dimensional vector
    
    train_final = pd.concat([train_ml.reset_index(drop=True),
                             train_emb_expanded.reset_index(drop=True)], axis=1)
    
    val_final = pd.concat([val_ml.reset_index(drop=True),
                           val_emb_expanded.reset_index(drop=True)], axis=1)
    
    print(train_final.shape)
    print(val_final.shape)
    # merging the original ml features with the expanded embeddings
    
    train_final_path = out_dir / "train_ml_final.parquet"
    val_final_path   = out_dir / "val_ml_final.parquet"
    
    train_final.to_parquet(train_final_path, index=False)
    val_final.to_parquet(val_final_path, index=False)
    
    print("saved:", train_final_path)
    print("saved:", val_final_path)
    # exporting the final datasets -> these files will be the inputs for notebook 5BIS
    
    print(train_final.head())
    # quick check everything worked fine
    
    # final dataset ready


if __name__ == "__main__":
    merge_features_with_embeddings()
