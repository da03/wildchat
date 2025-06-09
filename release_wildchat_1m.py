import torch
import sys
import json
import hashlib
from datetime import datetime
from datasets import Dataset
from collections import Counter
import tqdm

def most_common(l):
    c = Counter(l)
    value, count = c.most_common()[0]
    return value

#ds = torch.load(sys.argv[1])
ds = torch.load('arxiv-final-all-language-dialogue-output_1106_filtered_double_turns_may2-deid-analyze.pt')
#import pdb; pdb.set_trace()
#dataset = Dataset.from_dict({'conversation_id': conversation_ids, 'model': models, 'timestamp': timestamps, 'conversation': conversations, 'turn': turns, 'language': languages, 'openai_moderation': openai_moderations, 'detoxify_moderation': detoxify_moderations, 'toxic': toxics, 'redacted': redacteds})
#        assert set(d.keys()) == set(['model', 'timestamp', 'conversation', 'turn', 'language', 'toxic', 'state', 'country', 'hashed_ip', 'header']), d.keys()
toxics = ds['toxic']
models = ds['model']
states = ds['state']
countries = ds['country']
hashed_ips = ds['hashed_ip']
headers = ds['header']
turns = ds['turn']
languages = ds['language']
openai_moderations = [] #ds['openai_moderation']
detoxify_moderations = [] #ds['detoxify_moderation']
conversation_ids = []
conversations = ds['conversation']
timestamps = ds['timestamp']
redacteds = []
keys = set([])
total = 0
an = 0
for d in tqdm.tqdm(ds['conversation']):
    # turns
    if len(d) % 2 != 0:
        assert False
        print ('turn', len(d))
        continue
    if 'timestamp' not in d[-1]:
        assert False
        print ('timestamp')
        continue
    ## model
    #ms = []
    #for turn in d:
    #    if 'model' in turn:
    #        ms.append(turn['model'])
    #if len(ms) == 0:
    #    assert False
    #    continue
    #if not all([item == ms[-1] for item in ms]):
    #    assert False
    #    an += 1
    #    #continue
    #    #import pdb; pdb.set_trace()
    total += 1
    #model = ms[-1]
    #models.append(model)
    #turns.append(len(d) // 2)
    # language
    languages_ = []
    for turn in d:
        languages_.append(turn['language'].lower().capitalize())
    language = most_common(languages_)
    #languages.append(language)
    # openai moderations
    ms = []
    for turn in d:
        ms.append(turn['openai_moderation'])
    openai_moderations.append(ms)
    # detoxify results
    ms = []
    for turn in d:
        ms.append(turn['detoxify_moderation'])
    detoxify_moderations.append(ms)
    ## timestamps
    #ts = []
    #for i, turn in enumerate(d):
    #    if i % 2 == 1:
    #        ts.append(d['timestamp'])
    #timestamps.append(ts)
    # conversations
    cs = []
    r_overall = False
    toxic_overall = False
    cs_hash = []
    for i, turn in enumerate(d):
        r = False
        content = turn['content'].encode('utf-8', 'replace').decode()
        if '<PRESIDIO_ANONYMIZED_' in content:
            r = True
            r_overall = True
        #toxic = False
        #if turn['openai-toxicity']['flagged'] or turn['detoxify']['toxicity']>0.1:
        #    toxic = True
        #    toxic_overall = True
        #c = {'content': content, 'role': turn['role'], 'language': turn['language'].lower().capitalize(), 'redacted': r, 'toxic': toxic}
        #import pdb; pdb.set_trace()
        # dict_keys(['role', 'content', 'header', 'turn_identifier', 'language', 'openai_moderation', 'detoxify_moderation', 'toxic', 'country', 'state', 'hashed_ip', 'analyzer_results'])
        turn['redacted'] = r
        #c = {'content': content, 'role': turn['role'], 'language': turn['language'].lower().capitalize(), 'redacted': r, 'toxic': toxic}
        del turn['openai_moderation']
        del turn['detoxify_moderation']
        del turn['analyzer_results']
#        assert set(d.keys()) == set(['model', 'timestamp', 'conversation', 'turn', 'language', 'toxic', 'state', 'country', 'hashed_ip', 'header']), d.keys()
        #if i % 2 == 1:
        #    if 'timestamp' not in turn:
        #        an += 1
        #        print (an/total, an, total)
        #        #import pdb; pdb.set_trace()
        #if 'timestamp' in turn:
        #    c['timestamp'] = turn['timestamp']
        #cs.append(c)
        cs_hash.append({'content': content, 'role': turn['role']})
    #conversations.append(cs)
    timestamp = d[-1]['timestamp']
    key = hashlib.sha256(json.dumps(cs_hash).encode('utf-8')).hexdigest()[:32]
    if key in keys:
        print ('key', key)
    #assert key not in keys
    keys.add(key)
    conversation_ids.append(key)
    #toxics.append(toxic_overall)
    redacteds.append(r_overall)
    #timestamps.append(timestamp)
    #if len(keys) >= 100:
    #    break
    #break

    #if not toxic:
    #    toxic_free.append(d)

#key = hashlib.sha256(json.dumps({'prompt': prompt, 'model': model, 'temperature': temperature}).encode('utf-8')).hexdigest()
#import pdb; pdb.set_trace()
print (total)
dict_d = {'conversation_hash': conversation_ids, 'model': models, 'timestamp': timestamps, 'conversation': conversations, 'turn': turns, 'language': languages, 'openai_moderation': openai_moderations, 'detoxify_moderation': detoxify_moderations, 'toxic': toxics, 'redacted': redacteds, 'state': states, 'country': countries, 'hashed_ip': hashed_ips, 'header': headers}

#for k in dict_d:
#    dict_d[k] = dict_d[k][:100]
dataset = Dataset.from_dict(dict_d)
#    
#        assert set(d.keys()) == set(['model', 'timestamp', 'conversation', 'turn', 'language', 'toxic', 'state', 'country', 'hashed_ip', 'header']), d.keys()
#import pdb; pdb.set_trace()

from datasets import Value
#features = dataset.features
#features["timestamp"] = Value("timestamp[s, tz=UTC]")
#dataset = dataset.map(lambda ex: {"timestamp": datetime.strptime(ex["timestamp"], '%Y-%m-%d %H:%M:%S')}, features=features)

cutoff_date = datetime(2024, 4, 30, 0, 0, 0)
# Define the filter function
def filter_before_cutoff(example):
    example_date = example['timestamp']
    return example_date <= cutoff_date

dataset = dataset.filter(filter_before_cutoff)
#dataset = dataset.sort(['timestamp'])
print (len(dataset))
#import pdb; pdb.set_trace()
#dataset.push_to_hub('yuntian-deng/WildChat-1M-prerelease', split='train')
dataset.push_to_hub('allenai/WildChat-1M', split='train')
    

#ds = [{"conversation": l} for l in lst]
#out_file = open(sys.argv[1].replace('.pt', ''), "w")
#json.dump(ds, out_file, indent = 4)
#out_file.close()

#out_file = open(sys.argv[1].replace('.pt', '')+'-toxic-free', "w")
#ds = [{"conversation": l} for l in toxic_free]
#json.dump(ds, out_file, indent = 4)
#out_file.close()
print ('ms', an/total, an, total)
