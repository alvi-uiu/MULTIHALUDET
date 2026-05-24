import json
import os
import random
import warnings
from datasets import load_dataset

def load_halueval(lang='en'):
    """
    Loads the HaluEval dataset.
    For 'en', it attempts to load from a local file first, then falls back to HuggingFace.
    For other languages ('fr', 'bn', 'am'), it requires a local pre-translated JSON file.
    """
    local_path = f"data/halueval_{lang}.json"
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    if lang == 'en':
        print(f"Local {local_path} not found. Falling back to HuggingFace 'pminervini/HaluEval'.")
        dataset = load_dataset("pminervini/HaluEval", "qa_samples")
        samples = []
        for item in dataset['data']:
            samples.append({
                'question': item['question'],
                'answer': item['answer'],
                'is_hallucination': 1 if item['hallucination'] == 'yes' else 0
            })
        return samples
    else:
        raise FileNotFoundError(f"Translated dataset {local_path} not found. Please provide the translated dataset.")

def load_triviaqa(lang='en', seed=42, max_samples=10000):
    """
    Loads the TriviaQA dataset with Gemma-2-2B generated hard negatives.
    It requires a local JSON file containing the pre-generated hard negatives.
    """
    local_path = f"data/triviaqa_{lang}.json"
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            if max_samples and len(samples) > max_samples:
                random.seed(seed)
                samples = random.sample(samples, max_samples)
            return samples
            
    if lang == 'en':
        warnings.warn(
            f"Local {local_path} not found. The paper specifies using Gemma-2-2B to generate "
            "plausible hard negatives. Falling back to downloading TriviaQA from HuggingFace and "
            "using random sampling for negatives. Note: This contradicts the paper's methodology "
            "and is for testing pipeline execution only."
        )
        dataset = load_dataset("lucadiliello/triviaqa")
        all_items = list(dataset['train']) + list(dataset['validation'])
        
        if max_samples and len(all_items) > max_samples:
            random.seed(seed)
            all_items = random.sample(all_items, max_samples)

        samples = []
        for i, item in enumerate(all_items):
            question = item['question']
            correct_answer = item['answers'][0] if item['answers'] else ""
            if not correct_answer.strip():
                continue

            samples.append({
                'question': question,
                'answer': correct_answer,
                'is_hallucination': 0
            })

            wrong_idx = i
            while wrong_idx == i:
                wrong_idx = random.randint(0, len(all_items) - 1)
            wrong_answer = all_items[wrong_idx]['answers'][0] if all_items[wrong_idx]['answers'] else "unknown"

            samples.append({
                'question': question,
                'answer': wrong_answer,
                'is_hallucination': 1
            })

        random.seed(seed)
        random.shuffle(samples)
        return samples
    else:
        raise FileNotFoundError(f"Translated dataset {local_path} not found. Please provide the translated dataset with Gemma-2-2B negatives.")

