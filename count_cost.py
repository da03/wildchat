import tiktoken
import torch
from datetime import datetime
from collections import defaultdict
tokenizer = tiktoken.get_encoding("cl100k_base")
encoding = tokenizer
def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages."""
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value, disallowed_special=()))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
# Initialize a dictionary to store the data
token_counts = defaultdict(lambda: defaultdict(lambda: {'prompt_tokens': 0, 'response_tokens': 0}))
import pdb; pdb.set_trace()
import tqdm
for i in range(14):
    print ('IIIIII', i)
    dataset = torch.load(f'final.{i}.pt')
    # Process each turn in the dataset
    for turn in tqdm.tqdm(dataset):
        model = turn['model']
        timestamp = turn['timestamp']
        month_key = timestamp.strftime('%Y-%m')
    
        # Encode prompt and response
        prompt_tokens = num_tokens_from_messages(turn['payload']['messages'], model)
        response_tokens = len(tokenizer.encode(turn['partial_words'], disallowed_special=()))
    
        # Aggregate token counts
        token_counts[model][month_key]['prompt_tokens'] += prompt_tokens
        token_counts[model][month_key]['response_tokens'] += response_tokens
    def convert_defaultdict_to_dict(d):
        if isinstance(d, defaultdict):
            # Recursively convert defaultdict to dict
            d = {key: convert_defaultdict_to_dict(value) for key, value in d.items()}
        return d
    
    # Convert token_counts to a standard dictionary
    token_counts_standard = convert_defaultdict_to_dict(token_counts)
    torch.save(token_counts_standard, 'token_counts.pt')
    # Display or process the aggregated data
    for model, months in token_counts.items():
        for month, counts in months.items():
            print(f"Model: {model}, Month: {month}, Prompt Tokens: {counts['prompt_tokens']}, Response Tokens: {counts['response_tokens']}")
    
    from collections import defaultdict
    
    # Costs in dollars per million tokens for prompts and outputs
    model_costs = {
        'gpt-4-1106-preview': (10, 30),
        'gpt-4-turbo-2024-04-09': (10, 30),
        'gpt-4-0314': (30, 60),
        'gpt-4-0125-preview': (10, 30),
        'gpt-3.5-turbo-0613': (1.5, 2),
        'gpt-3.5-turbo-0301': (1.5, 2),
        'gpt-3.5-turbo-0125': (0.5, 1.5)
    }
    
    # Convert token counts to costs
    def calculate_cost(model, token_counts):
        if model not in model_costs:
            print (model)
            return 0
        prompt_cost, response_cost = model_costs[model]
        total_prompt_cost = (token_counts['prompt_tokens'] / 1e6) * prompt_cost
        total_response_cost = (token_counts['response_tokens'] / 1e6) * response_cost
        return total_prompt_cost + total_response_cost
    
    # Assuming token_counts is the dictionary you calculated previously
    total_costs = defaultdict(float)
    total_tokens = defaultdict(lambda: {'prompt_tokens': 0, 'response_tokens': 0})
    monthly_costs = defaultdict(lambda: defaultdict(float))
    
    for model, months in token_counts.items():
        for month, counts in months.items():
            cost = calculate_cost(model, counts)
            monthly_costs[month][model] = cost
            total_costs[model] += cost
            total_tokens[model]['prompt_tokens'] += counts['prompt_tokens']
            total_tokens[model]['response_tokens'] += counts['response_tokens']
    
    # Display monthly costs, total costs, and total tokens
    for month, models in monthly_costs.items():
        for model, cost in models.items():
            print(f"Month: {month}, Model: {model}, Cost: ${cost:.2f}")
    
    for model, cost in total_costs.items():
        prompt_tokens = total_tokens[model]['prompt_tokens']
        response_tokens = total_tokens[model]['response_tokens']
        print(f"Total Cost for Model {model}: ${cost:.2f}, Total Prompt Tokens: {prompt_tokens}, Total Response Tokens: {response_tokens}")
    
    # Calculate overall total cost
    overall_total_cost = sum(total_costs.values())
    print(f"Overall Total Cost: ${overall_total_cost:.2f}")
