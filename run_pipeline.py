import argparse
import torch
import numpy as np
import random
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm import tqdm

from src.config import get_config, MODEL_REGISTRY
from src.data.loader import load_halueval, load_triviaqa
from src.data.feature_extractor import extract_features
from src.training.trainer import train_deep_model_fold, extract_features_batch
from src.ensemble.meta_learner import get_ensemble, calibrate_ensemble, temperature_scale
from src.utils.metrics import find_best_thresholds, evaluate_all, compute_uncertainty_metrics
from src.utils.visualization import plot_publication_results

def stage_1_extract(args, config):
    print(f"\n--- Stage 1: Dynamic Layer Probing & Feature Extraction ---")
    print(f"Dataset: {args.dataset} ({args.lang}), Model: {args.model}")
    
    if args.dataset == "halueval":
        samples = load_halueval(lang=args.lang)
    else:
        samples = load_triviaqa(lang=args.lang, seed=config.seed)
        
    config.model_name = MODEL_REGISTRY[args.model]
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_llm = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.model_dtype,
        device_map="auto"
    )
    model_llm.eval()
    
    all_seq, all_glob, all_labels = [], [], []
    for i, sample in enumerate(tqdm(samples)):
        try:
            s, g = extract_features(sample['question'], sample['answer'], tokenizer, model_llm, config)
            all_seq.append(s)
            all_glob.append(g)
            all_labels.append(sample['is_hallucination'])
            if i > 0 and i % 500 == 0:
                torch.cuda.empty_cache()
        except Exception as e:
            continue

    X_seq = np.nan_to_num(np.array(all_seq), nan=0.0)
    X_glob = np.nan_to_num(np.array(all_glob), nan=0.0)
    y = np.array(all_labels)
    
    os.makedirs('results/features', exist_ok=True)
    np.save(f'results/features/X_seq_{args.dataset}_{args.lang}_{args.model}.npy', X_seq)
    np.save(f'results/features/X_glob_{args.dataset}_{args.lang}_{args.model}.npy', X_glob)
    np.save(f'results/features/y_{args.dataset}_{args.lang}_{args.model}.npy', y)
    
    del model_llm
    torch.cuda.empty_cache()
    gc.collect()
    print("Stage 1 completed. Features saved.")

def stage_2_3_train_oof(args, config):
    print(f"\n--- Stage 2 & 3: Multi-Scale Modeling & Out-of-Fold Generation ---")
    try:
        X_seq = np.load(f'results/features/X_seq_{args.dataset}_{args.lang}_{args.model}.npy')
        X_glob = np.load(f'results/features/X_glob_{args.dataset}_{args.lang}_{args.model}.npy')
        y = np.load(f'results/features/y_{args.dataset}_{args.lang}_{args.model}.npy')
    except FileNotFoundError:
        print("Features not found. Please run --stage 1 first.")
        return
        
    X_seq_train, X_seq_test, X_glob_train, X_glob_test, y_train, y_test = train_test_split(
        X_seq, X_glob, y, test_size=config.test_size, stratify=y, random_state=config.seed
    )
    
    N, L, F_seq = X_seq_train.shape
    scaler_seq = RobustScaler()
    X_seq_train = scaler_seq.fit_transform(X_seq_train.reshape(-1, F_seq)).reshape(N, L, F_seq)
    X_seq_test = scaler_seq.transform(X_seq_test.reshape(-1, F_seq)).reshape(X_seq_test.shape[0], L, F_seq)
    
    scaler_glob = RobustScaler()
    X_glob_train = scaler_glob.fit_transform(X_glob_train)
    X_glob_test = scaler_glob.transform(X_glob_test)
    
    skf = StratifiedKFold(n_splits=config.n_inner_folds, shuffle=True, random_state=config.seed)
    oof_features = np.zeros((len(y_train), config.hidden_dim + (config.hidden_dim // 2)))
    test_features_accum = np.zeros((len(y_test), config.hidden_dim + (config.hidden_dim // 2)))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_seq_train, y_train)):
        print(f"Fold {fold+1}/{config.n_inner_folds}")
        model, _ = train_deep_model_fold(
            X_seq_train[tr_idx], X_glob_train[tr_idx], y_train[tr_idx],
            X_seq_train[val_idx], X_glob_train[val_idx], y_train[val_idx],
            config
        )
        
        feats_val = extract_features_batch(model, X_seq_train[val_idx], X_glob_train[val_idx])
        oof_features[val_idx] = feats_val
        test_features_accum += extract_features_batch(model, X_seq_test, X_glob_test)
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    X_train_deep = oof_features
    X_test_deep = test_features_accum / config.n_inner_folds
    
    os.makedirs('results/oof', exist_ok=True)
    np.save(f'results/oof/X_train_deep_{args.dataset}_{args.lang}_{args.model}.npy', X_train_deep)
    np.save(f'results/oof/X_test_deep_{args.dataset}_{args.lang}_{args.model}.npy', X_test_deep)
    np.save(f'results/oof/y_train_{args.dataset}_{args.lang}_{args.model}.npy', y_train)
    np.save(f'results/oof/y_test_{args.dataset}_{args.lang}_{args.model}.npy', y_test)
    print("Stage 2 & 3 completed. OOF features saved.")

def stage_4_ensemble(args, config):
    print(f"\n--- Stage 4: Log-Odds Ensemble Meta-Learning ---")
    try:
        X_train_deep = np.load(f'results/oof/X_train_deep_{args.dataset}_{args.lang}_{args.model}.npy')
        X_test_deep = np.load(f'results/oof/X_test_deep_{args.dataset}_{args.lang}_{args.model}.npy')
        y_train = np.load(f'results/oof/y_train_{args.dataset}_{args.lang}_{args.model}.npy')
        y_test = np.load(f'results/oof/y_test_{args.dataset}_{args.lang}_{args.model}.npy')
    except FileNotFoundError:
        print("OOF features not found. Please run --stage 2 first.")
        return
        
    scaler_deep = StandardScaler()
    X_train_deep = scaler_deep.fit_transform(X_train_deep)
    X_test_deep = scaler_deep.transform(X_test_deep)
    
    ensemble = get_ensemble(config.seed)
    calibrated_model, _ = calibrate_ensemble(ensemble, X_train_deep, y_train)
    
    probs = calibrated_model.predict_proba(X_test_deep)[:, 1]
    
    thresholds = find_best_thresholds(probs, y_test)
    results = evaluate_all(probs, y_test, thresholds['youden'])
    uncertainty = compute_uncertainty_metrics(probs, y_test, (probs >= thresholds['youden']).astype(int))
    
    print("\nResults Summary:")
    print(f"AUC: {results['auc']:.4f}, F1: {results['f1']:.4f}, Acc: {results['acc']:.4f}")
    
    os.makedirs('results/plots', exist_ok=True)
    plot_path = f"results/plots/results_{args.dataset}_{args.lang}_{args.model}.png"
    plot_publication_results(
        probs, y_test, (probs >= thresholds['youden']).astype(int), 
        uncertainty, results['auc'], results['ap'], results['cm'], 
        plot_path
    )
    print(f"Saved results to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="MultiHaluDet 4-Stage Execution Pipeline")
    parser.add_argument("--dataset", type=str, default="halueval", choices=["halueval", "triviaqa"])
    parser.add_argument("--lang", type=str, default="en", choices=["en", "fr", "bn", "am"], help="Language variant of the dataset")
    parser.add_argument("--model", type=str, default="mistral-7b", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--stage", type=str, default="all", choices=["all", "extract", "train_oof", "ensemble"], 
                        help="all: Run full pipeline; extract: Feature Extraction; train_oof: Multi-Scale Modeling & OOF; ensemble: Ensemble Meta-Learning")
    args = parser.parse_args()
    
    config = get_config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    if args.stage in ["extract", "all"]:
        stage_1_extract(args, config)
    if args.stage in ["train_oof", "all"]:
        stage_2_3_train_oof(args, config)
    if args.stage in ["ensemble", "all"]:
        stage_4_ensemble(args, config)

if __name__ == "__main__":
    main()
