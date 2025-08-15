from datasets import load_dataset
from collections import Counter
import re

def group_model_name(name):
    # Custom grouping rules
    if name.startswith("gpt-4o-"):
        return "gpt-4o"
    elif name.startswith("gpt-4-turbo-"):
        return "gpt-4-turbo"
    elif name.startswith("gpt-4-"):
        return "gpt-4"
    elif name.startswith("gpt-3.5-turbo-"):
        return "gpt-3.5-turbo"
    elif name.startswith("o1-preview-"):
        return "o1-preview"
    elif name.startswith("o1-mini-"):
        return "o1-mini"
    else:
        # fallback: take the part before first date-like suffix
        return re.split(r"-\d{4}-\d{2}-\d{2}", name)[0]

def count_grouped(repo_id, split="train"):
    ds = load_dataset(repo_id, split=split)
    grouped_counts = Counter(group_model_name(m) for m in ds["model"])
    print(f"\nGrouped counts for {repo_id}:")
    for family, count in grouped_counts.most_common():
        print(f"{family}: {count:,}")
    print(f"Total examples: {len(ds):,}")

# Example runs
#count_grouped("allenai/WildChat-4.8M-Full")
count_grouped("allenai/WildChat-4.8M")

