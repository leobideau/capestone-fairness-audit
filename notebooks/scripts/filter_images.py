import os
import shutil
import pandas as pd

# paths
RAW_TRAIN = "data/raw/train"
RAW_VAL = "data/raw/val"

BAL_TRAIN_CSV = "data/processed/balanced_train.csv"
BAL_VAL_CSV = "data/processed/balanced_val.csv"

OUT_TRAIN = "data/processed/balanced_images/train"
OUT_VAL = "data/processed/balanced_images/val"


def copy_split(df, raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    copied = 0
    missing = 0

    for f in df["file"]:
        filename = os.path.basename(f)
        src = os.path.join(raw_dir, filename)
        dst = os.path.join(out_dir, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

    print(f"{out_dir} -> copied: {copied}, missing: {missing}")


def main():
    train_df = pd.read_csv(BAL_TRAIN_CSV)
    val_df = pd.read_csv(BAL_VAL_CSV)

    print("Copying train...")
    copy_split(train_df, RAW_TRAIN, OUT_TRAIN)

    print("Copying val...")
    copy_split(val_df, RAW_VAL, OUT_VAL)

    print("Done.")


if __name__ == "__main__":
    main()
#This script takes the balanced CSVs, finds the matching images in the raw folders,
#and copies only those into a clean balanced_images/ folder.
#This gives me a lightweight dataset aligned with my balanced labels.
