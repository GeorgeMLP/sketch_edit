import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

def create_subset(input_path, output_path, percent=0.1, seed=42):
    """
    Creates a subset of the dataset by sampling a percentage of data from ALL splits
    evenly from each category (task), with a fixed random seed for reproducibility.
    """
    random.seed(seed)
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    new_data = {
        'metadata': data.get('metadata', {}),
        'train': [],
        'val': [],
        'test': []
    }
    
    for split in ['train', 'val', 'test']:
        split_data = data.get(split, [])
        if not split_data:
            continue
            
        # Group by task
        task_groups = defaultdict(list)
        for item in split_data:
            task_groups[item['task']].append(item)
        
        # Sample from each task
        subset = []
        for task, items in task_groups.items():
            num_to_sample = int(len(items) * percent)
            if len(items) >= num_to_sample:
                sampled = random.sample(items, num_to_sample)
            else:
                sampled = items
                print(f"Warning: Split '{split}', Task '{task}' has only {len(items)} samples, taking all.")
            subset.extend(sampled)
        
        new_data[split] = subset
        print(f"Split '{split}': Sampled {len(subset)} items across {len(task_groups)} tasks.")

    # Update metadata
    new_data['metadata']['num_samples_subset'] = {k: len(v) for k, v in new_data.items() if isinstance(v, list)}
    new_data['metadata']['subset_percent'] = percent
    new_data['metadata']['subset_seed'] = seed
    
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"Successfully created subset and saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample a percentage of the dataset evenly across tasks for all splits.")
    parser.add_argument("--input", type=str, required=True, help="Path to the full dataset.json")
    parser.add_argument("--output", type=str, required=True, help="Path to save the sampled dataset_subset.json")
    parser.add_argument("--percent", type=float, default=0.1, help="Fraction to sample (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    create_subset(
        input_path=args.input,
        output_path=args.output,
        percent=args.percent,
        seed=args.seed
    )
