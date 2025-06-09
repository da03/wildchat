from datasets import load_dataset

# Load the dataset in streaming mode
dataset = load_dataset("yuntian-deng/WildChat-4M-Full-Internal", split="train", streaming=True)

# Print the first 5 examples
for idx, example in enumerate(dataset):
    if idx >= 5:
        break
    print(f"Example {idx + 1}:")
    print(example)
    print("-" * 50)
