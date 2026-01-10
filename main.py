"""
Main pipeline for FairFace bias analysis project
Executes all notebooks in order
"""

from pathlib import Path
import sys
import importlib.util


def import_from_file(filepath, module_name):
    """Helper to import a module from a .py file"""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    """Execute full pipeline"""
    
    BASE_PATH = Path(__file__).parent
    scripts_dir = BASE_PATH / "src"
    
    # load all my modules
    mod01 = import_from_file(scripts_dir / "01_data_preparation.py", "data_prep")
    mod02 = import_from_file(scripts_dir / "02_baseline_predictions.py", "baseline_pred")
    mod02bis = import_from_file(scripts_dir / "02BIS_extract_embeddings.py", "embeddings")
    mod03 = import_from_file(scripts_dir / "03_baseline_evaluation.py", "baseline_eval")
    mod04 = import_from_file(scripts_dir / "04_feature_engineering.py", "features")
    mod04bis = import_from_file(scripts_dir / "04BIS_merge_embeddings.py", "merge_emb")
    mod05 = import_from_file(scripts_dir / "05_ML_error_prediction.py", "ml_baseline")
    mod05bis = import_from_file(scripts_dir / "05BIS_predicting_with_embeddings.py", "ml_final")
    mod06 = import_from_file(scripts_dir / "06_results.py", "results")
    mod07 = import_from_file(scripts_dir / "07_cases_analysis.py", "cases")

    print("fairface bias analysis pipeline")

    try:
        print("\n[1/10] Data preparation: balancing dataset...")
        mod01.prepare_and_balance_data()
        print("[DONE] Step 1 complete\n")
        
        print("[2/10] Baseline predictions: running DeepFace...")
        mod02.run_deepface_predictions()
        print("[DONE] Step 2 complete\n")
        
        print("[3/10] Extracting embeddings: Facenet512")
        mod02bis.extract_embeddings()
        print("[DONE] Step 3 complete\n")
        
        print("[4/10] Baseline evaluation: measuring DeepFace performance")
        mod03.evaluate_baseline()
        print("[DONE] Step 4 complete\n")
        
        print("[5/10] Feature engineering: extracting image features")
        mod04.engineer_features()
        print("[DONE] Step 5 complete\n")
        
        print("[6/10] Merging embeddings with features")
        mod04bis.merge_features_with_embeddings()
        print("[DONE] Step 6 complete\n")
        
        print("[7/10] Training ML models without embeddings")
        mod05.train_ml_baseline()
        print("[DONE] Step 7 complete\n")
        
        print("[8/10] Training ML models with embeddings + XGBoost tuning")
        mod05bis.train_ml_with_embeddings()
        print("[DONE] Step 8 complete\n")
        
        print("[9/10] Generating final results and visualizations")
        mod06.generate_results()
        print("[DONE] Step 9 complete\n")
        
        print("[10/10] Analyzing specific cases")
        mod07.analyze_cases()
        print("[DONE] Step 10 complete\n")
        

        print("pipeline complete:")
        print("\nResults saved in:")
        print("  - data/processed/")
        print("  - data/predictions/")
        print("  - data/embeddings/")
        print("  - data/ml_ready/")
        print("  - data/ml_final/")
        print("  - results/")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
