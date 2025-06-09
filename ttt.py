from datasets import Dataset, Sequence, Features, Value

# Define a function to generate a sample with "tools" feature
def generate_sample():
    # Generate random sample data
    sample_data = {
        "text": "Sample text",
        "feature_1": []
    }
    
    # Add feature_1 with random keys for this sample
    feature_1 = [{"key1": "value1"}, {"key1": "value2"}]  # Example feature_1 with random keys
    sample_data["feature_1"].extend(feature_1)
    
    return sample_data

# Generate multiple samples
num_samples = 10
samples = [generate_sample() for _ in range(num_samples)]

def gen():
    for item in samples:
        yield item
# Create a Hugging Face Dataset
features = Features({
    'text':              Value('string'),
    'feature_1':       Sequence(Features({
        'key1':          Value('string')}))})
features = Features({
    'text':              Value('string'),
    'feature_1':       [Features({
        'key1':          Value('string')})]})
#dataset = Dataset.from_generator(gen, features=features)
dataset = Dataset.from_list(samples)#, features=features)
print (dataset.features)
print (dataset[0])
