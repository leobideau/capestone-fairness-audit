def prepare_and_balance_data():
    # pivot 1 : fairface data exploration, data audit, cleaning and rebalancing
    # the goal is to prepare and balance the fairface dataset by race and gender for bias analysis and model training

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os
    import shutil

    # Get the root directory
    BASE_PATH = Path(__file__).parent.parent

    # first thing is locating where the FairFace csv files are in the project
    train_csv = BASE_PATH / "data" / "raw" / "fairface_label_train.csv"
    val_csv = BASE_PATH / "data" / "raw" / "fairface_label_val.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print("Train set :")
    print(train_df.head())

    print("\nValidation set :")
    print(val_df.head())
    # then I loaded train and validation csvs with demographic labels nd checked everything was ok

    train_df["split"] = "train"
    val_df["split"] = "val"

    full_df = pd.concat([train_df, val_df], ignore_index=True)

    print(full_df[["race", "gender", "age"]].describe(include="all"))
    # I merged the train and validation sets to get an overview of the full dataset and check the distributions of race, gender, and age to see if balancing was needed

    for col in ["race", "gender", "age"]:
        counts = full_df[col].value_counts().sort_index()
        plt.figure(figsize=(8, 3))
        counts.plot(kind="bar")
        plt.title(col)
        plt.ylabel("count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    # reload clean data for processing
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # configuration
    RANDOM_SEED = 42
    TARGET_TRAIN_PER_GROUP = 500
    TARGET_VAL_PER_GROUP = 150

    def balance_by_race_gender(df, target_per_group, random_state=RANDOM_SEED):
        """balanced dataset for genre and ethnicity."""
        grouped = df.groupby(["race", "gender"], group_keys=False)
        return grouped.apply(
            lambda g: g.sample(
                n=min(target_per_group, len(g)),
                random_state=random_state
            )
        )

    balanced_train = balance_by_race_gender(train_df, TARGET_TRAIN_PER_GROUP)
    balanced_val = balance_by_race_gender(val_df, TARGET_VAL_PER_GROUP)

    print("Balanced train size:", len(balanced_train))
    print(balanced_train[["race", "gender"]].value_counts().sort_index())

    print("\nBalanced validation size:", len(balanced_val))
    print(balanced_val[["race", "gender"]].value_counts().sort_index())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    balanced_train["race"].value_counts().sort_index().plot(kind="bar", ax=axes[0])
    axes[0].set_title("Balanced train set")
    axes[0].set_xlabel("race")
    axes[0].set_ylabel("count")

    balanced_val["race"].value_counts().sort_index().plot(kind="bar", ax=axes[1])
    axes[1].set_title("Balanced validation set")
    axes[1].set_xlabel("race")

    plt.tight_layout()
    plt.show()

    # create output directory
    output_dir = BASE_PATH / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # save balanced datasets
    balanced_train.to_csv(output_dir / "balanced_train.csv", index=False)
    balanced_val.to_csv(output_dir / "balanced_val.csv", index=False)
    print("saved")

    # copy balanced images
    RAW_TRAIN = BASE_PATH / "data" / "raw" / "train"
    RAW_VAL = BASE_PATH / "data" / "raw" / "val"

    OUT_TRAIN = BASE_PATH / "data" / "processed" / "balanced_images" / "train"
    OUT_VAL = BASE_PATH / "data" / "processed" / "balanced_images" / "val"

    def copy_split(df, raw_dir, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        missing = 0

        for f in df["file"]:
            filename = os.path.basename(f)
            src = raw_dir / filename
            dst = out_dir / filename

            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1

        print(f"{out_dir} -> copied: {copied}, missing: {missing}")

    print("\nCopying train images...")
    copy_split(balanced_train, RAW_TRAIN, OUT_TRAIN)

    print("Copying val images...")
    copy_split(balanced_val, RAW_VAL, OUT_VAL)

    # Check if images exist
    train_path = BASE_PATH / "data" / "processed" / "balanced_images" / "train"
    val_path = BASE_PATH / "data" / "processed" / "balanced_images" / "val"

    if train_path.exists():
        print("train images ok :", len(os.listdir(train_path)), "images")
    else:
        print("train images not ok ")

    if val_path.exists():
        print("val images ok :", len(os.listdir(val_path)), "images")
    else:
        print("val images not ok ")

    # Check missing files
    missing_train = [f for f in balanced_train["file"] if not (train_path / os.path.basename(f)).exists()]
    missing_val = [f for f in balanced_val["file"] if not (val_path / os.path.basename(f)).exists()]

    print("Missing in train:", len(missing_train))
    print("Missing in val:", len(missing_val))


if __name__ == "__main__":
    prepare_and_balance_data()
