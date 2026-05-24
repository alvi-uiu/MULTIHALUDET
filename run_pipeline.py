import argparse
import torch
import numpy as np
import random
import gc
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="halueval", choices=["halueval", "triviaqa"])
    parser.add_argument("--model", type=str, default="mistral-7b", choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()
    
    config = get_config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "halueval":
        samples = load_halueval()
    else:
        samples = load_triviaqa(seed=config.seed)
        
    print(f"Loading {args.model} model...")
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
    print("Extracting features...")
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
    
    del model_llm
    torch.cuda.empty_cache()
    gc.collect()
    
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
    oof_features = None
    test_features_accum = None
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_seq_train, y_train)):
        print(f"Fold {fold+1}/{config.n_inner_folds}")
        model, _ = train_deep_model_fold(
            X_seq_train[tr_idx], X_glob_train[tr_idx], y_train[tr_idx],
            X_seq_train[val_idx], X_glob_train[val_idx], y_train[val_idx],
            config
        )
        
        feats_val = extract_features_batch(model, X_seq_train[val_idx], X_glob_train[val_idx])
        if oof_features is None:
            oof_features = np.zeros((len(y_train), feats_val.shape[1]))
            test_features_accum = np.zeros((len(y_test), feats_val.shape[1]))
        
        oof_features[val_idx] = feats_val
        test_features_accum += extract_features_batch(model, X_seq_test, X_glob_test)
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    X_train_deep = oof_features
    X_test_deep = test_features_accum / config.n_inner_folds
    
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
    
    plot_publication_results(
        probs, y_test, (probs >= thresholds['youden']).astype(int), 
        uncertainty, results['auc'], results['ap'], results['cm'], 
        f"results_{args.dataset}_{args.model}.png"
    )
    print(f"Saved results to results_{args.dataset}_{args.model}.png")

if __name__ == "__main__":
    main()
