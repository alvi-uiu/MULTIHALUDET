import random
from datasets import load_dataset

def load_halueval():
    dataset = load_dataset("pminervini/HaluEval", "qa_samples")
    samples = []
    for item in dataset['data']:
        samples.append({
            'question': item['question'],
            'answer': item['answer'],
            'is_hallucination': 1 if item['hallucination'] == 'yes' else 0
        })
    return samples

def load_triviaqa(seed=42, max_samples=10000):
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
