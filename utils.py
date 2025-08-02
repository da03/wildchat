import sys
import itertools
import shutil
import glob
import copy
import re
import os
import gc
import sqlite3
import torch
from datasets import load_dataset, Dataset, Value, load_from_disk, Features, Sequence
from huggingface_hub import create_repo
#from lingua import LanguageDetectorBuilder
from multiprocessing import Process, Queue, cpu_count
from multiprocessing import Pool
import requests
import sys
import yaml
import time
import numpy as np
import tqdm
import hashlib
import json
import mysql.connector
from datetime import datetime

from openai import OpenAI
import openai
from detoxify import Detoxify
from collections import Counter
from tenacity import (
retry,
wait_exponential,
wait_fixed,
stop_after_attempt,
)
import hashlib
import os
# At the top-level of your file:


def check_record(record, features, path=""):
    if isinstance(features, Features):
        assert isinstance(record, dict), f"Expected dict at {path}, got {type(record)}"
        for key in record:
            if key not in features:
                print (record[key])
                raise AssertionError(f"Unexpected key '{path + key}' found in record but missing in features.")
            check_record(record[key], features[key], path=f"{path}{key}.")
    elif isinstance(features, Sequence) or isinstance(features, list):
        assert isinstance(record, list), f"Expected list at {path}, got {type(record)}"
        for idx, item in enumerate(record):
            check_record(item, features.feature if hasattr(features, 'feature') else features[0], path=f"{path}[{idx}].")
    elif isinstance(features, Value):
        pass
    else:
        raise AssertionError(f"Unhandled schema type at {path}: {features}")

def hash_ip_with_salt(ip_address, salt):
    """Hashes an IP address with a given salt using SHA-256."""
    salted_ip = salt + ip_address.encode()  # Combine the salt and the IP address
    hashed_ip = hashlib.sha256(salted_ip).hexdigest()
    return hashed_ip

def save_salt(salt, filepath):
    """Saves the salt to a file."""
    with open(filepath, 'w') as file:
        file.write(salt.hex())  # Write the hexadecimal representation of the salt

def load_salt(filepath):
    """Loads the salt from a file."""
    with open(filepath, 'r') as file:
        hex_salt = file.read()
    return bytes.fromhex(hex_salt)  # Convert back to bytes

salt_file_path = 'salt.txt'

read_salt_flag = True

if read_salt_flag:
    # Load the salt from file
    batch_salt = load_salt(salt_file_path)
else:
    assert False
    # Save the salt to file for future use
    save_salt(batch_salt, salt_file_path)



def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

credential_file = '/home/wentingz/credential.yaml'
if os.path.exists(credential_file):
    with open(credential_file, 'r') as f:
        config = yaml.safe_load(f)
    
    headers = {
        'Authorization': config["key"],
        'OpenAI-Organization': config["org"]
    }

def extract_header(headers):
    return {k: headers[k] for k in ['user-agent', 'accept-language'] if k in headers}

def get_fingerprint(headers):
    # Concatenate selected header values
    #fingerprint_source = "".join([headers['user-agent'], headers['accept-language'], headers['x-forwarded-for'].split(',')[0]])
    fingerprint_source = "".join([headers['user-agent'] if 'user-agent' in headers else '', headers['accept-language'] if 'accept-language' in headers else ''])
    
    # Hash the concatenated string
    fingerprint_hash = hashlib.sha256(fingerprint_source.encode('utf-8')).hexdigest()
    
    return fingerprint_hash

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_user_input(filename):
    save_name = 'user-' + filename.split('/')[-1].replace('.csv', '.pt')
    if os.path.exists(save_name):
        outs = torch.load(save_name)
    else:
        data = load(filename)
        outs = set()
        for d in data:
            for one in d["payload"]["messages"]:
                if one["role"] == "user":
                    outs.add(one["content"])
        outs = sorted(outs)
        torch.save(outs, save_name)
    return outs


database_name_moderation = './cache/openai_responses.db'
table_name_moderation = 'openai_responses'
database_name_detoxify = './cache/detoxify_responses.db'
table_name_detoxify = 'detoxify_responses'
database_name_language = './cache/language_responses.db'
table_name_language = 'language_responses'

def create_database(database_name, table_name):
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    # Create table
    c.execute(f'''CREATE TABLE IF NOT EXISTS {table_name}
                 (key TEXT PRIMARY KEY, prompt TEXT, completion TEXT)''')
    conn.commit()
    conn.close()

create_database(database_name_moderation, table_name_moderation)
create_database(database_name_detoxify, table_name_detoxify)
create_database(database_name_language, table_name_language)

@retry(wait=wait_fixed(1), stop=stop_after_attempt(7))
def insert_or_update(database_name, table_name, key, prompt, completion):
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    c.execute(f'''INSERT OR REPLACE INTO {table_name}
                 (key, prompt, completion) VALUES (?, ?, ?)''', 
                 (key, prompt, completion))
    conn.commit()
    conn.close()

def retrieve(database_name, table_name, key):
    conn = sqlite3.connect(database_name)
    c = conn.cursor()

    c.execute(f"SELECT prompt, completion FROM {table_name} WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    if result:
        return (True, result)
    else:
        return (False, None)


def query_f_with_cache(database_name, table_name, f, content):
    key = hashlib.sha256(content.encode('utf-8')).hexdigest()
    hit, result = retrieve(database_name, table_name, key)
    #if hit:
    #    prompt, output = result
    #    output = json.loads(output)
    #    if '_splitted' in output:
    #        hit = False
    #        print ('splitted')
    if not hit:
        output, finished = f(content)
        #print ('success')
        if finished:
            insert_or_update(database_name, table_name, key, content, json.dumps(output))
    else:
        #print ('hit')
        #print (content)
        prompt, output = result
        output = json.loads(output)
    return output, hit

#@retry(wait=wait_fixed(10), stop=stop_after_attempt(7))
def query_moderation(content, max_num_retries=2, wait_time=20, raise_e=False):
    print ('ca', content[:10], len(content))
    client = OpenAI()
    num_retries = 0
    finished = False
    while (not finished) and num_retries <= max_num_retries:
        if num_retries > 0:
            print (f'retrying {num_retries} times')
        try:
            response = client.moderations.create(model="omni-moderation-latest", input=content)
            finished = True
        except Exception as e:
            if raise_e:
                raise e
            err_msg = f'{e}'
            print (err_msg)
            m = re.search(r"Please try again in (\d+\.?\d*)s", err_msg)
            num_retries += 1
            if m:
                sleep_time = min(float(m.group(1)) * 1.2, wait_time)
                sleep_time = wait_time
                print (f'sleeping: {sleep_time}')
                time.sleep(sleep_time)
            else:
                time.sleep(wait_time)
    if not finished:
        #import pdb; pdb.set_trace()
        content_length = len(content)
        half_length = int(round(content_length / 2))
        content_firsthalf = content[:half_length]
        content_secondhalf = content[half_length:]
        print (f'splitting, old length: {content_length}, new length: {half_length}')
        #output_firsthalf = query_moderation(content_firsthalf, max_num_retries, wait_time)
        output_firsthalf, _ = query_f_with_cache(database_name_moderation, table_name_moderation, query_moderation, content_firsthalf)
        #output_secondhalf = query_moderation(content_secondhalf, max_num_retries, wait_time)
        output_secondhalf, _ = query_f_with_cache(database_name_moderation, table_name_moderation, query_moderation, content_secondhalf)
        output = {'flagged': output_firsthalf['flagged'] or output_secondhalf['flagged']}
        output['categories'] = {}
        for k in output_firsthalf['categories']:
            output['categories'][k] = output_firsthalf['categories'][k] or output_secondhalf['categories'][k]
        output['category_scores'] = {}
        for k in output_firsthalf['category_scores']:
            output['category_scores'][k] = max(output_firsthalf['category_scores'][k], output_secondhalf['category_scores'][k])
        output['_splitted'] = True
        #import pdb; pdb.set_trace()
    else:
        output = response.results[0].model_dump()
    #return output, finished
    return output, True 

#global detector
#detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
def query_language_with_cache_worker(contents):
    from lingua import LanguageDetectorBuilder
    detector_local = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    #detector_local = copy.deepcopy(detector_local)
    def query_language(content):
        print ('ca', content[:10], len(content))
        confidence_values = detector_local.compute_language_confidence_values(content)
        language = str(confidence_values[0].language).split('.')[-1]
        return language, True 
    i = 0
    contents = list(contents)
    for content in tqdm.tqdm(contents):
        if i % 1000 == 0:
            print (i)
        try:
            query_f_with_cache(database_name_language, table_name_language, query_language, content)
        except Exception as e:
            print (f'Skipping Error: {e}')
            #time.sleep(1)
        i += 1

def query_moderation_with_cache_worker(contents):
    i = 0
    contents = list(contents)
    for content in tqdm.tqdm(contents):
        if i % 1000 == 0:
            print (i)
        try:
            query_f_with_cache(database_name_moderation, table_name_moderation, query_moderation, content)
        except Exception as e:
            print (f'Skipping Error: {e}')
            time.sleep(1)
        i += 1

def remove_wildbench(save_name):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')] # todo
    # gather all chunk files
    files = glob.glob(f'{save_name}.cacheddict.withlang.chunk*.pt')
    # sort by numeric index after “chunk”
    def chunk_index(path):
        # this regex finds the digits between “chunk” and “.pt”
        m = re.search(r'chunk(\d+)\.pt$', path)
        return int(m.group(1)) if m else None
    files = sorted(files, key=chunk_index)
    for chunk_idx, file in enumerate(files):
        print (f'loading {file}')
        d = torch.load(file, weights_only=False)
        first_50char_to_id = {}
        for i, conversation in enumerate(d['conversation']):
            first_50char = conversation[0]['content'][:50]
            if first_50char not in first_50char_to_id:
                first_50char_to_id[first_50char] = []
            first_50char_to_id[first_50char].append(i)

        wildbench_ids = json.load(open('wildbench_ids.json'))
        #import pdb; pdb.set_trace()
        missing = 0
        ids_to_remove = []
        for example in wildbench_ids:
            first_50char = example['conversation_1st_turn_50chars']
            if first_50char not in first_50char_to_id:
                print ('d')
                missing += 1
                continue
            ids = first_50char_to_id[first_50char]
            if len(ids) == 0:
                #import pdb; pdb.set_trace()
                print ('a')
            elif len(ids) > 1:
                #import pdb; pdb.set_trace()
                print ('b')
                ids_new = []
                for i in ids:
                    try:
                        timestamp = datetime.strptime(example['timestamp'], '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)
                    except Exception:
                        timestamp = datetime.strptime(example['timestamp'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=None)
                    if d['conversation'][i][-1]['timestamp'] == timestamp:
                        ids_new.append(i)
                ids = ids_new
                if len(ids) == 0:
                    #import pdb; pdb.set_trace()
                    print ('ba')
                if len(ids) > 1:
                    #import pdb; pdb.set_trace()
                    print ('bb')
            #assert len(ids) == 1
            ids_to_remove.extend(ids)
        ids_to_remove = set(ids_to_remove)
        print (missing, len(ids_to_remove))
        keys = d.keys()
        for key in keys:
            values_new = []
            values = d[key]
            for i, val in enumerate(tqdm.tqdm(values)):
                if i in ids_to_remove:
                    continue
                values_new.append(val)
            d[key] = values_new
        torch.save(d, f'{save_name}.cacheddict.withlang.rmwildbench.chunk{chunk_idx}.pt')
        for key in d:
            print (key, len(d[key]))
        conversations = d['conversation']
        print (len(conversations))
        print (sum ([len(item) for item in conversations]))





def add_moderation(save_name):
    GENERATE_DATABASE = True
    GENERATE_DATABASE = False
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')] # todo
    # gather all chunk files
    files = glob.glob(f'{save_name}.cacheddict.withlang.rmwildbench.chunk*.pt')
    # sort by numeric index after “chunk”
    def chunk_index(path):
        # this regex finds the digits between “chunk” and “.pt”
        m = re.search(r'chunk(\d+)\.pt$', path)
        return int(m.group(1)) if m else None
    files = sorted(files, key=chunk_index)
    for chunk_idx, file in enumerate(files):
        if GENERATE_DATABASE:
            print (f'loading {file}')
            d = torch.load(file, weights_only=False)
            conversations = d['conversation'] # TODO: sort by date, remove last day
            #languages = []
            queries = []
            for conversation in tqdm.tqdm(conversations):
                #langs = []
                for turn in conversation:
                    if turn["content"] == "":
                        pass
                    else:
                        queries.append(turn['content'])
            del d
            #queries = queries[:10]
            #import pdb; pdb.set_trace()
            #n = 50
            n = 37
            n = 128
            #n = 32
            n = 63
            n = 11
            n = 13
            n = 11
            n = 7
            n = 23
            n = 37
            #import pdb; pdb.set_trace()
            #query_moderation_with_cache_worker(queries[:10])
            #import pdb; pdb.set_trace()
            with Pool(n) as pool:
                pool.map(query_moderation_with_cache_worker, chunks(queries, n))
            print ('pooled moderation')
            #sys.exit(1)
    import pdb; pdb.set_trace()
    for chunk_idx, file in enumerate(files):
        print (f'loading {file}')
        d = torch.load(file, weights_only=False)
        conversations = d['conversation'] # TODO: sort by date, remove last day
        import collections
        c = collections.defaultdict(int)
        aaa = 0
        english = 0
        total = 0
        total_c = 0
        counts = collections.defaultdict(int)
        for conversation in tqdm.tqdm(conversations):
            for turn in conversation:
                content = turn["content"]
                if content == "":
                    results = {}
                else:
                    results, _ = query_f_with_cache(database_name_moderation, table_name_moderation, query_moderation, content)
                if turn['language'].lower() == 'english':
                    english += 1
                counts[turn['language']] += 1
                total += 1
                if '_splitted' in results:
                    total_c += 1
                    del results['_splitted']
                    #print (turn.keys())
                    print (turn['language'], english/total)
                    c[turn['language']] += 1
                    aaa += 1
                    #with open(f'examples/{turn["language"]}_{c[turn["language"]]}.txt', 'w') as fout:
                    #    fout.write(content)
                    print (c)
                turn['openai_moderation'] = results
        torch.save(d, f'{save_name}.cacheddict.withlang.rmwildbench.moderations.chunk{chunk_idx}.pt')
        #counts = [(-counts[k], k) for k in counts]
        #counts = sorted(counts)
        #with open('counts.txt', 'w') as fout:
        #    for v, k in counts:
        #        fout.write(f'{k}: {-v/ total}\n')
        #counts = [(-c[k], k) for k in c]
        #counts = sorted(counts)
        #with open('c.txt', 'w') as fout:
        #    for v, k in counts:
        #        fout.write(f'{k}: {-v/ total_c}\n')
        #print (counts)



def query_detoxify_with_cache_worker(args):
    worker_id, contents = args
    print (f'Creating detoxify with cuda id {worker_id}')
    model = Detoxify('multilingual', device=f"cuda:{worker_id}")
    def query_detoxify(content):
        results = model.predict(content)
        for k in results:
            v = results[k]
            if type(v).__module__ == np.__name__:
                results[k] = v.item()
        return results, True

    i = 0
    for content in tqdm.tqdm(list(contents)):
        #if i % 1000 == 0:
        #    print (i)
        try:
            query_f_with_cache(database_name_detoxify, table_name_detoxify, query_detoxify, content)
        except Exception as e:
            print ('error', f'{e}')
        i += 1


def add_detoxify(save_name):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')] # todo
    # gather all chunk files
    files = glob.glob(f'{save_name}.cacheddict.withlang.rmwildbench.moderations.chunk*.pt')
    # sort by numeric index after “chunk”
    def chunk_index(path):
        # this regex finds the digits between “chunk” and “.pt”
        m = re.search(r'chunk(\d+)\.pt$', path)
        return int(m.group(1)) if m else None
    files = sorted(files, key=chunk_index)
    GENERATE_DATABASE = False
    GENERATE_DATABASE = True
    for chunk_idx, file in enumerate(files):
        if GENERATE_DATABASE:
            print (f'loading {file}')
            d = torch.load(file, weights_only=False)
            conversations = d['conversation'] # TODO: sort by date, remove last day
            queries = []
            for conversation in tqdm.tqdm(conversations):
                for turn in conversation:
                    if turn["content"].strip() == "":
                        pass
                        #results = {}
                    else:
                        queries.append(turn['content'])#results = model.predict(turn['content'])
            print (len(queries))
            n = torch.cuda.device_count()
            print(f"Number of available GPUs: {n}")
            if n == 1:
                query_detoxify_with_cache_worker((0, queries))
            else:
                #import pdb; pdb.set_trace()
                with Pool(n) as pool:
                    pool.map(query_detoxify_with_cache_worker, enumerate(chunks(queries, n)))
            #sys.exit(1)
    import pdb; pdb.set_trace()
    sys.exit(1)
    worker_id = 0
    #model = Detoxify('multilingual', device=f"cpu")
    model = Detoxify('multilingual', device=f"cuda:{worker_id}")
    def query_detoxify(content):
        results = model.predict(content)
        for k in results:
            v = results[k]
            if type(v).__module__ == np.__name__:
                results[k] = v.item()
        return results, True

    for conversation in tqdm.tqdm(conversations):
        for turn in conversation:
            content = turn["content"]
            if content == "":
                results = {}
            else:
                results, _ = query_f_with_cache(database_name_detoxify, table_name_detoxify, query_detoxify, content)
            turn['detoxify_moderation'] = results
    torch.save(d, f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.pt')

def hash_ips(save_name):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')] # todo

    d = torch.load(f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.pt') #TODO: use lang
    conversations = d['conversation'] # TODO: sort by date, remove last day

    import geoip2.database

    with geoip2.database.Reader('/mnt/wildchat/natural-dialogues/GeoLite2-City_20240416/GeoLite2-City.mmdb') as reader:
        def get_state_country(ip):
            try:
                response = reader.city(ip)
                country = response.country.name
                state = response.subdivisions.most_specific.name
            except Exception as e:
                state = None
                country = None
            return state, country

        #import pdb; pdb.set_trace()
        toxics = []
        for conversation in tqdm.tqdm(conversations):
            conversation_toxic = False
            for i, turn in enumerate(conversation):
                turn_toxic = False
                if turn['content'] != '' and (turn['openai_moderation']['flagged'] or turn['detoxify_moderation']['toxicity'] > 0.1):
                    turn_toxic = True
                    conversation_toxic = True
                turn['toxic'] = turn_toxic

                if i % 2 == 0:
                    ip = turn["ip"]
                    state, country = get_state_country(ip)
                    turn['country'] = country
                    turn['state'] = state
                    del turn['ip']
                    turn['hashed_ip'] = hash_ip_with_salt(ip, batch_salt)
                else:
                    assert 'ip' not in turn
            toxics.append(conversation_toxic)
        states = []
        countries = []
        hashed_ips = []
        for ip in d['ip']:
            state, country = get_state_country(ip)
            states.append(state)
            countries.append(country)
            hashed_ips.append(hash_ip_with_salt(ip, batch_salt))
        d['state'] = states
        d['country'] = countries
        d['toxic'] = toxics
        del d['ip']
        d['hashed_ip'] = hashed_ips
        del d['device_fingerprint']

        a = d.keys()
        assert set(d.keys()) == set(['model', 'timestamp', 'conversation', 'turn', 'language', 'toxic', 'state', 'country', 'hashed_ip', 'header']), d.keys()
        torch.save(d, f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.pt')

def push_dataset(save_name):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')]
    files = glob.glob(f'{save_name}.cacheddict.withlang.chunk*.pt')
    # sort by numeric index after “chunk”
    def chunk_index(path):
        # this regex finds the digits between “chunk” and “.pt”
        m = re.search(r'chunk(\d+)\.pt$', path)
        return int(m.group(1)) if m else None
    files = sorted(files, key=chunk_index)
    filtered_records = []
    cutoff = datetime(2025, 8, 1)  # filter out from Aug 2025 onwards
    # explicit schema for nested structs
    features = Features({
    'model':              Value('string'),
    'timestamp':          Value('timestamp[us]'),
    'conversation':       [Features({
        'content':          Value('string'),
        'created':          Value('int64'),
        'header':           Features({
            'accept-language': Value('string'),
            'user-agent':      Value('string'),
        }),
        'ip':               Value('string'),
        'language':         Value('string'),
        'openai_id':        Value('string'),
        'role':             Value('string'),
        'temperature':      Value('float64'),
        'timestamp':        Value('timestamp[us]'),
        'token_counter':    Value('int64'),
        'top_p':            Value('float64'),
        'turn_identifier':  Value('int64'),
        'system_fingerprint': Value('string'),
        'usage': Features({
            'completion_tokens':         Value('int64'),
            'completion_tokens_details': Features({
                'reasoning_tokens': Value('int64'),
                'text_tokens': Value('int64'),
                'audio_tokens': Value('int64'),
                'accepted_prediction_tokens': Value('int64'),
                'rejected_prediction_tokens': Value('int64')
            }),
            'prompt_tokens':             Value('int64'),
            'total_tokens':              Value('int64'),
            'prompt_tokens_details': Features({
                'cached_tokens': Value('int64'),
                'audio_tokens': Value('int64'),
            }),
        }),
    })],
    'turn':               Value('int64'),
    'ip':                 Value('string'),
    'device_fingerprint': Value('string'),
    'header':             Features({
        'accept-language': Value('string'),
        'user-agent':      Value('string'),
    }),
    'language':           Value('string'),
})

    print("Loading and filtering records...")
    def record_generator():
        for path in files:
            print(f"  → loading {path}")
            chunk = torch.load(path, weights_only=False)
            keys = list(chunk.keys())
            n = len(chunk[keys[0]])
            for i in range(n):
                record = {k: chunk[k][i] for k in keys}
                check_record(record, features)
                if record['timestamp'] < cutoff:
                    yield record
            # free memory for this chunk before moving on
            del chunk
            gc.collect()

    #sample = list(itertools.islice(record_generator(), 1000))
    #print(f"🔢 Sampled {len(sample)} records to infer schema…")
    #ds_sample = Dataset.from_list(sample)
    #features = ds_sample.features
    #print("👀 Inferred features:", features)
    #import pdb; pdb.set_trace()
    ds = Dataset.from_generator(record_generator, features=features)

    ## 1) write real Arrow shards to disk
    #export_dir = "./hf_export"
    ## clear out any old export
    #shutil.rmtree(export_dir, ignore_errors=True)
    #ds_stream.save_to_disk(export_dir)

    ## 2) load it back as a regular Dataset
    #ds = load_from_disk(export_dir)
    #import pdb; pdb.set_trace()
    #print(f"🔢 Dataset contains {ds.num_rows} examples.")

    # 3) ensure the repo exists, then push the full shards
    repo_size_str = f"{len(ds)/1e6:.1f}M"
    repo_id = f"yuntian-deng/WildChat-{repo_size_str}-Full-Internal"
    #repo_id = "yuntian-deng/WildChat-4M-Full-Internal"
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    ## Push to Hugging Face
    ds.push_to_hub(repo_id, split='train')
    print(f"✅ Full dataset pushed to https://hf.co/{repo_id}")

    
def add_languages(save_name):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')]
    #d = torch.load(f'{save_name}.cacheddict.pt')
    # gather all chunk files
    files = glob.glob(f'{save_name}.cacheddict.chunk*.pt')
    # sort by numeric index after “chunk”
    def chunk_index(path):
        # this regex finds the digits between “chunk” and “.pt”
        m = re.search(r'chunk(\d+)\.pt$', path)
        return int(m.group(1)) if m else None
    files = sorted(files, key=chunk_index)
    
    ## Initialize an empty dict to hold combined data
    #d = {}
    
    for chunk_idx, file in enumerate(files):
        print (f'loading {file}')
        d = torch.load(file, weights_only=False)
        conversations = d['conversation'] # TODO: sort by date, remove last day
        queries = []
        for conversation in tqdm.tqdm(conversations):
            for turn in conversation:
                if turn["content"].strip() == "":
                    pass
                else:
                    queries.append(turn['content'])
        from lingua import LanguageDetectorBuilder
        detector_local = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
        confidence_values_list = detector_local.compute_language_confidence_values_in_parallel(queries)
        languages = []
        i = 0
        for conversation in tqdm.tqdm(conversations):
            langs = []
            for turn in conversation:
                if turn["content"].strip() == "":
                    language = 'nolang'
                else:
                    #confidence_values = detector.compute_language_confidence_values(turn["content"])
                    confidence_values = confidence_values_list[i]
                    i += 1
                    language = str(confidence_values[0].language).split('.')[-1]
                language = language.title()
                langs.append(language)
                turn["language"] = language
            languages.append(most_frequent(langs))
        d['language'] = languages
        total_size = len(languages)
        for key in d:
            assert len(d[key]) == total_size, (len(d[key]), key)
    
        torch.save(d, f'{save_name}.cacheddict.withlang.chunk{chunk_idx}.pt')
        print(f'Saved chunk {chunk_idx}')

    #import pdb; pdb.set_trace()
    #dataset = Dataset.from_dict(d)
    #dataset.push_to_hub('allenai/internal_WildChat', split='train')

def link_conversations(save_name):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')]
    #d = {'model': all_models, 'conversation': all_conversations, 'turn': turns, 'ip': all_ips, 'device_fingerprint': all_device_fingerprints, 'timestamp': all_timestamps}
    #torch.save(d, f'{save_name}.cacheddict.pt')
    #d = torch.load(f'{save_name}.cacheddict.pt')
    #import pdb; pdb.set_trace()
    #for conversation in d['conversation']:
    #    for turn in conversation:
    #        turn['content'] = turn['content'].encode('utf-8', errors='ignore').decode('utf-8')
    #import pdb; pdb.set_trace()

    #dataset = Dataset.from_dict(d)
    #dataset.push_to_hub('allenai/internal_WildChat', split='train')
    #sys.exit(1)
    

    #a = torch.load('dumps/wildchat.feb14.2024.0.pt')
    #items = [item for item in a if item['device_fingerprint'] == 'c9c31eb82d042e1f34ead91b676fdc34928e7bd23daeb49744638d4266ba2af4']
    #import pdb; pdb.set_trace()
    filename = f'{save_name}.hash.pt'
    times = []
    results = torch.load(filename, weights_only=False)
    # earlier dates first
    results = sorted(results, key=lambda x: x['timestamp'])
    current_turn_hashes = {}
    max_delta_timestamp = None
    #max_delta_seconds = 3600*6 # maximum 6 hours
    total_nolink = 0
    total_processed = 0
    #import pdb; pdb.set_trace()
    for result_id, result in enumerate(results):
        ip = result['ip']
        device_fingerprint = result['device_fingerprint']
        timestamp = result['timestamp']
        model = result['model']
        flag_nolink = False 
        result['is_final'] = True
        if result['prev_turn_len'] > 0:
            # link to a previous msg
            prev_turn_hash = result['prev_turn_hash']

            if prev_turn_hash not in current_turn_hashes:
                flag_nolink = True
            else:
                #import pdb; pdb.set_trace()
                #link_candidates = list(filter(lambda x: x[2] == device_fingerprint and x[1] == ip and x[4] == model and results[x[0]]['is_final'] and (timestamp - x[3]).total_seconds()<max_delta_seconds, current_turn_hashes[prev_turn_hash]))
                link_candidates = list(filter(lambda x: x[2] == device_fingerprint and x[1] == ip and x[4] == model and results[x[0]]['is_final'], current_turn_hashes[prev_turn_hash]))
                if len(link_candidates) == 0:
                    #link_candidates = list(filter(lambda x: x[2] == device_fingerprint and x[4] == model and results[x[0]]['is_final'] and (timestamp - x[3]).total_seconds()<max_delta_seconds, current_turn_hashes[prev_turn_hash]))
                    link_candidates = list(filter(lambda x: x[2] == device_fingerprint and x[4] == model and results[x[0]]['is_final'], current_turn_hashes[prev_turn_hash]))
                #link_candidates = list(filter(lambda x: x[2] == device_fingerprint and x[4] == model and results[x[0]]['is_final'] and (timestamp - x[3]).total_seconds()<max_delta_seconds, link_candidates))
                if len(link_candidates) == 0:
                    flag_nolink = True
                else:
                    prev_turn_id = link_candidates[-1][0]
                    result['prev_turn_id'] = prev_turn_id
                    results[prev_turn_id]['is_final'] = False
                    results[prev_turn_id]['pointed_by'] = result_id
                    delta_timestamp = timestamp - link_candidates[-1][3]
                    times.append(delta_timestamp.total_seconds())
                    #if delta_timestamp.total_seconds() > 3600*24*10:
                    #    import pdb; pdb.set_trace()
                    #    a = torch.load('dumps/wildchat.feb14.2024.0.pt')
                    #    b = torch.load('dumps/wildchat.feb14.2024.1.pt')
                    if max_delta_timestamp is None:
                        max_delta_timestamp = delta_timestamp
                    else:
                        max_delta_timestamp = max(max_delta_timestamp, delta_timestamp)
        current_turn_hash = result['current_turn_hash']
        if current_turn_hash not in current_turn_hashes:
            current_turn_hashes[current_turn_hash] = []
        current_turn_hashes[current_turn_hash].append((result_id, ip, device_fingerprint, timestamp, model))
        total_processed += 1
        if flag_nolink:
            total_nolink += 1
            print (total_nolink / total_processed, total_nolink, total_processed, max_delta_timestamp)
    #import pdb; pdb.set_trace()
    import numpy as np
    times = np.array(times)
    for p in [1, 2, 5, 10, 50, 80, 90, 95, 98, 99]:
        print (f'p: {p}, percentile: {np.percentile(times, p)}')
    #import pdb; pdb.set_trace()
    #a = torch.load('dumps/wildchat.feb14.2024.0.pt')
    #line = line.encode('latin1').decode('unicode-escape').encode('latin1').decode('utf8')
    all_results = []

    all_conversations = []
    all_ips = []
    all_device_fingerprints = []
    all_models = []
    all_timestamps = []
    all_headers = []
    turns = []

    chunk_id_result_id_to_conversation_id = {}
    for result_id, result in enumerate(results):
        conversation = []
        if 'is_final' in result and result['is_final']:
            current_result = result
            conversation.append(result_id)
            flag_invalid = False
            while current_result['prev_turn_len'] > 0:
                if 'prev_turn_id' not in current_result:
                    flag_invalid = True
                    break
                prev_turn_id = current_result['prev_turn_id']
                current_result = results[prev_turn_id]
                conversation.append(prev_turn_id)
            if flag_invalid:
                continue
            conversation = conversation[::-1]
            turns.append(len(conversation))
            infos = [results[turn_id] for turn_id in conversation]
            assert len(infos) > 0, len(infos)
            for info_id, info in enumerate(infos):
                chunk_id = info['chunk_id']
                turn_position = info['turn_position']
                if (chunk_id, turn_position) not in chunk_id_result_id_to_conversation_id:
                    chunk_id_result_id_to_conversation_id[(chunk_id, turn_position)] = []
                chunk_id_result_id_to_conversation_id[(chunk_id, turn_position)].append([len(all_results), info_id])
                assert len(chunk_id_result_id_to_conversation_id[(chunk_id, turn_position)]) == 1
            all_results.append(infos)
            all_conversations.append([[] for _ in range(2*len(conversation))])
            all_models.append(result['model'])
            all_timestamps.append(result['timestamp'])
            all_ips.append(None)
            all_headers.append(None)
            #all_headers.append(result['headers'])
            #all_ips.append(result['ip'])
            all_device_fingerprints.append(result['device_fingerprint'])
    print (f'total: {len(all_models)}')
    chunk_id = 0
    filename = f'{save_name}.{chunk_id}.pt'
    total_processed = 0
    wrong = 0
    #import pdb; pdb.set_trace()
    while os.path.exists(filename):
        print (f'parsing {filename}')
        chunk = torch.load(filename, weights_only=False)
        for turn_position, turn in enumerate(tqdm.tqdm(chunk)):
            if (chunk_id, turn_position) in chunk_id_result_id_to_conversation_id:
                for position, info_id in chunk_id_result_id_to_conversation_id[(chunk_id, turn_position)]:
                    turn_header = extract_header(turn['headers'])
                    all_conversations[position][info_id*2] = {'role': 'user', 'content': turn['payload']['messages'][-1]['content'].encode('utf-8', errors='ignore').decode('utf-8'), 'ip': turn['ip'], 'header': turn_header, 'turn_identifier': turn['turn_id']}
                    all_conversations[position][info_id*2+1] = {'role': 'assistant', 'content': turn['partial_words'].encode('utf-8', errors='ignore').decode('utf-8'), 'timestamp': turn['timestamp'], 'turn_identifier': turn['turn_id']}
                    if 'usage' in turn:
                        #import pdb; pdb.set_trace()
                        all_conversations[position][info_id*2+1]['usage'] = turn['usage']
                    for k in ['token_counter', 'id', 'created', 'system_fingerprint']:
                        if k in turn:
                            k_tgt = k
                            if k == 'id':
                                k_tgt = 'openai_id'
                            all_conversations[position][info_id*2+1][k_tgt] = turn[k]
                    for k in ['temperature', 'top_p']:
                        if k in turn['payload']:
                            all_conversations[position][info_id*2+1][k] = turn['payload'][k]
                        else:
                            import pdb; pdb.set_trace()
                            print ("dfsdfs")
                    for k in turn:
                        if k not in all_conversations[position][info_id*2+1] and k not in all_conversations[position][info_id*2] and k not in ['payload', 'partial_words', 'counter', 'id', 'model', 'headers', 'turn_id', 'device_fingerprint', 'object', 'service_tier']:
                            import pdb; pdb.set_trace()
                            print (k)
                    assert all_models[position] == turn['model']
                    assert all_timestamps[position] >= turn['timestamp']
                    if all_ips[position] is None:
                        all_ips[position] = []
                    all_ips[position].append(turn['ip'])
                    if all_headers[position] is None:
                        all_headers[position] = []
                    all_headers[position].append(json.dumps(turn_header))
        chunk_id += 1
        filename = f'{save_name}.{chunk_id}.pt'
    #import pdb; pdb.set_trace()
    for i in range(len(all_headers)):
        all_headers[i] = json.loads(most_frequent(all_headers[i]))
    #import pdb; pdb.set_trace()
    for i in range(len(all_ips)):
        all_ips[i] = most_frequent(all_ips[i])
    #import pdb; pdb.set_trace()
    #d = {'model': all_models, 'conversation': all_conversations, 'turn': turns, 'ip': all_ips, 'device_fingerprint': all_device_fingerprints, 'timestamp': all_timestamps, 'header': all_headers}
    d = {'model': all_models, 'timestamp': all_timestamps, 'conversation': all_conversations, 'turn': turns, 'ip': all_ips, 'device_fingerprint': all_device_fingerprints, 'header': all_headers}
    #torch.save(d, f'{save_name}.cacheddict.pt')
    chunk_size = 200000  # Adjust this size based on memory constraints
    total_size = len(all_models)
    for key in d:
        assert len(d[key]) == total_size, (len(d[key]), key)
    
    for chunk_start in range(0, total_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_size)
        
        chunk_dict = {key: d[key][chunk_start:chunk_end] for key in d}
        
        chunk_idx = chunk_start // chunk_size
        torch.save(chunk_dict, f'{save_name}.cacheddict.chunk{chunk_idx}.pt')
        print(f'Saved chunk {chunk_idx}: items {chunk_start} to {chunk_end}')

    #del d['device_fingerprint']
    #import pdb; pdb.set_trace()

    #dataset.push_to_hub('allenai/internal_WildChat', split='train')
    


def generate_hash(save_name):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')]
    chunk_id = 0
    results = []
    filename = f'{save_name}.{chunk_id}.pt'
    #import pdb; pdb.set_trace()
    total_processed = 0
    wrong = 0
    while os.path.exists(filename):
        print (f'parsing {filename}')
        chunk = torch.load(filename, weights_only=False)
        for turn_position, turn in enumerate(tqdm.tqdm(chunk)):
            result = {
                    'chunk_id': chunk_id,
                    'filename': filename,
                    'turn_position': turn_position,
                    'ip': turn['ip'],
                    'timestamp': turn['timestamp'],
                    'device_fingerprint': turn['device_fingerprint'],
                    'model': turn['model'],
                    }
            msgs = turn['payload']['messages']
            total_processed += 1
            flag_wrong = False
            for msg_id, msg in enumerate(msgs):
                if msg_id % 2 == 0:
                    if msg['role'] != 'user':
                        flag_wrong = True
                        break
                elif msg_id % 2 == 1:
                    if msg['role'] != 'assistant':
                        flag_wrong = True
                        break
            if flag_wrong:
                wrong += 1
                print ('wrong', wrong/total_processed, wrong, total_processed)
                continue
            assert msgs[-1]['role'] == 'user', msgs[-1]['role']
            msgs = msgs + [{"content": turn["partial_words"], "role": "assistant"}]
            msgs_json = json.dumps(msgs, sort_keys=True)
            msgs_hash = hashlib.sha256(msgs_json.encode('utf-8')).hexdigest()
            result['current_turn_hash'] = msgs_hash
            msgs = msgs[:-2]
            prev_turn_len = len(msgs)
            #if prev_turn_len % 2 != 0:
            #    import pdb; pdb.set_trace()
            assert prev_turn_len % 2 == 0
            prev_turn_len = prev_turn_len // 2
            result['prev_turn_len'] = prev_turn_len 
            if prev_turn_len > 0:
                #import pdb; pdb.set_trace()
                msgs_json = json.dumps(msgs, sort_keys=True)
                msgs_hash = hashlib.sha256(msgs_json.encode('utf-8')).hexdigest()
                result['prev_turn_hash'] = msgs_hash
            results.append(result)
        chunk_id += 1
        filename = f'{save_name}.{chunk_id}.pt'
    print ('wrong', wrong/total_processed, wrong, total_processed)
    torch.save(results, f'{save_name}.hash.pt')

def load_mysql(save_name):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')]
    USERNAME = os.getenv('MYSQL_USERNAME')
    PASSWORD = os.getenv('MYSQL_PASSWORD')
    DATABASE = os.getenv('MYSQL_DATABASE')

    # Define your connection parameters
    config = {
        'user': USERNAME,
        'password': PASSWORD,
        'host': 'localhost',
        'database': DATABASE,
        'raise_on_warnings': True,
    }

    # Connect to the database
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor(dictionary=True)

        new_timeout = 28800  # Example: 8 hours in seconds
        cursor.execute(f"SET SESSION wait_timeout = {new_timeout}")
        cursor.execute(f"SET SESSION net_read_timeout = {new_timeout}")
        cursor.execute(f"SET SESSION net_write_timeout = {new_timeout}")
        #cursor.execute(f"SET SESSION connect_timeout = {new_timeout}")
        cnx.commit()


        #query = "SELECT * FROM main WHERE date > '2023-04-10 00:00:00'"
        #query = "SELECT * FROM main WHERE date > '2023-04-08 22:01:49'"
        query = "SELECT * FROM main WHERE date > '2023-04-09 00:00:00'"
        #query = "SHOW VARIABLES LIKE '%timeout%';"
        #query = "SELECT @@GLOBAL.MAX_EXECUTION_TIME, @@SESSION.MAX_EXECUTION_TIME;"
        cursor.execute(query)

        ### Fetch all the rows
        #rows = cursor.fetchall()
        #print (rows)
        #sys.exit(1)

        # Fetch rows in chunks
        chunk_size = 200000
        rows = cursor.fetchmany(chunk_size)
        chunk_id = 0
        processed = 0
        last_date = None
        while rows:
            results = []
            #import pdb; pdb.set_trace()
            for row in rows:
                turn_id = row['id']
                date = row['date']
                if last_date is not None:
                    if not date >= last_date:
                        print ('TIMESTAMP VIOLATION')
                        print (last_date, date)
                last_date = date
                text = row['text']
                content = text.lower()
                if 'yuntian' in content:
                    content = content.replace('yuntian could', '')
                    content = content.replace('yuntian-deng', '')
                    content = content.replace('yuntian d', '')
                    if 'yuntian' in content:
                        print ('skipping')
                        if 'this is a test from yuntian' not in content:
                            print ('-'*10)
                            print (content)
                            print ('='*10)
                        continue

                result = json.loads(text)
                if 'headers' not in result:
                    print ('heaer')
                    continue
                try:
                    result['headers'] = eval(result['headers'])
                except Exception as e:
                    print (e)
                    continue
                fingerprint = get_fingerprint(result['headers'])

                ip_address = result['headers']['x-forwarded-for'].split(',')[0]
                result['ip'] = ip_address
                result['turn_id'] = turn_id
                result['timestamp'] = date
                result['device_fingerprint'] = fingerprint
                results.append(result)
                processed += 1
                if processed % 10000 == 0:
                    print (f'processed: {processed}')
            torch.save(results, save_name + f'.{chunk_id}.pt')
            rows = cursor.fetchmany(chunk_size)
            chunk_id += 1
            print (f'processed: {processed}')

    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))

    finally:
        if cnx.is_connected():
            cursor.close()
            cnx.close()


def load(filename):
    save_name = filename.split('/')[-1].replace('.csv', '.pt')
    outs = []
    true = True
    false = False
    if os.path.exists(save_name):
        outs = torch.load(save_name)
    else:
        with open(filename, 'r') as f:
            for line in f:
                #import pdb; pdb.set_trace()
                if 'x-forwarded-for' in line:
                    ip_start = line.find("'x-forwarded-for': '") + len("'x-forwarded-for': '")
                    ip_end = line.find("', 'x-forwarded-proto'")
                    ip = line[ip_start:ip_end]
                else:
                    ip = "N/A"
                start = line.find(',')
                end = line.rfind(',')
                timestamp = line[end+1:].replace("\"", "").strip()
                line = line[start+2:end-1]
                line = line.encode('latin1').decode('unicode-escape').encode('latin1').decode('utf8')
                line = eval(line)
                line['timestamp'] = timestamp
                line['ip'] = ip
                outs.append(line)
        torch.save(outs, save_name)
    return outs

def prepare_first_turn_user_data(name):
    if ".pt" in name:
        dialogues = torch.load(name)
        data = [dialogue[0]["content"] for dialogue in dialogues if dialogue[0]["role"] == "user"]
        data = [d.encode('utf-8', 'replace').decode() for d in data]
    elif name == "databricks/databricks-dolly-15k":
        ds = load_dataset(name, split="train")
        data = []
        for d in ds:
            if d["context"] == "":
                data.append(d["instruction"])
            else:
               data.append(d["context"] + " " + d["instruction"])
    elif name == "OpenAssistant/oasst1":
        ds = load_dataset(name, split="train")
        data = [d["text"] for d in ds if d["role"] == "prompter" and d["parent_id"] is None]
        ds = load_dataset(name, split="validation")
        data += [d["text"] for d in ds if d["role"] == "prompter" and d["parent_id"] is None]
    elif name == "tatsu-lab/alpaca":
        ds = load_dataset(name, split="train")
        data = []
        for d in ds:
            if d["input"] == "":
                data.append(d["instruction"])
            else:
                data.append(d["instruction"] + d["input"])
    elif name == "sharegpt":
        dialogues = load_dataset('json', data_files="data/ShareGPT_V3_unfiltered_cleaned_split.json", split='train')
        data = [dialogue["conversations"][0]["value"] for dialogue in dialogues if len(dialogue["conversations"])>0 and dialogue["conversations"][0]["from"] == "human"]
    else:
        NotImplementedError
    return data

def prepare_complete_user_data(name):
    if ".pt" in name:
        dialogues = torch.load(name)
        data = [turn["content"] for dialogue in dialogues for turn in dialogue if turn["role"] == "user"]
        data = [d.encode('utf-8', 'replace').decode() for d in data]
    elif name == "databricks/databricks-dolly-15k":
        ds = load_dataset(name, split="train")
        data = []
        for d in ds:
            if d["context"] == "":
                data.append(d["instruction"])
            else:
               data.append(d["context"] + " " + d["instruction"])
    elif name == "OpenAssistant/oasst1":
        ds = load_dataset(name, split="train")
        data = [d["text"] for d in ds if d["role"] == "prompter"]
        ds = load_dataset(name, split="validation")
        data += [d["text"] for d in ds if d["role"] == "prompter"]
    elif name == "tatsu-lab/alpaca":
        ds = load_dataset(name, split="train")
        data = []
        for d in ds:
            if d["input"] == "":
                data.append(d["instruction"])
            else:
                data.append(d["instruction"] + d["input"])
    elif name == "sharegpt":
        dialogues = load_dataset('json', data_files="data/ShareGPT_V3_unfiltered_cleaned_split.json", split='train')
        data = [turn["value"] for dialogue in dialogues for turn in dialogue["conversations"] if turn["from"] == "human"]
    else:
        NotImplementedError
    return data

def prepare_response_data(name):
    if name == "databricks/databricks-dolly-15k":
        ds = load_dataset(name, split="train")
        data = []
        for d in ds:
            data.append(d["response"])
    elif name == "OpenAssistant/oasst1":
        ds = load_dataset(name, split="train")
        data = [d["text"] for d in ds if d["role"] == "assistant"]
        ds = load_dataset(name, split="validation")
        data += [d["text"] for d in ds if d["role"] == "assistant"]
    elif name == "tatsu-lab/alpaca":
        ds = load_dataset(name, split="train")
        data = []
        for d in ds:
            data.append(d["output"])
    elif name == "sharegpt":
        dialogues = load_dataset('json', data_files="data/ShareGPT_V3_unfiltered_cleaned_split.json", split='train')
        data = [turn["value"] for dialogue in dialogues for turn in dialogue["conversations"] if turn["from"] == "gpt"]
    else:
        NotImplementedError
    return data

def convert_format(name):
    if ".pt" in name:
        ds = torch.load(name)
        data = []
        for d in ds:
            for turn in d:
                if turn["language"] != "ENGLISH": break
                turn['content'] = turn['content'].encode('utf-8', 'replace').decode()
            else:
                data.append(d)
    elif name == "databricks/databricks-dolly-15k":
        ds = load_dataset(name, split="train")
        data = []
        for d in ds:
            data.append([{"role": "user", "content": d["context"] + " " + d["instruction"]}, {"role": "assistant", "content": d["response"]}])
    elif name == "OpenAssistant/oasst1":
        ds = [x for x in load_dataset("OpenAssistant/oasst1", split="validation")] + [x for x in load_dataset("OpenAssistant/oasst1", split="train")]
        parents = set([x['parent_id'] for x in ds])
        messages = set([x['message_id'] for x in ds])
        leaves = [x for x in messages if x not in parents]
        data = []
        d = {one['message_id']: one for one in ds}
        for leave in leaves:
            convs = []
            while True:
                if d[leave]['parent_id'] is not None:
                    convs = [d[leave]] + convs
                    leave = d[leave]['parent_id']
                else:
                    convs = [d[leave]] + convs
                    break
            convs = [{"role": "user" if turn["role"] == "prompter" else "assistant", "content": turn["text"], "language": turn["lang"]} for turn in convs]
            data.append(convs)
            #languages = [turn["language"] != "en" for turn in convs]
            #if sum(languages) == 0:
            #    data.append(convs)
    elif name == "tatsu-lab/alpaca":
        ds = load_dataset(name, split="train")
        data = []
        for d in ds:
            data.append([{"role": "user", "content": d["instruction"] + d["input"]}, {"role": "assistant", "content": d["output"]}])
    elif name == "sharegpt":
        ds = load_dataset('json', data_files="data/ShareGPT_V3_unfiltered_cleaned_split.json", split='train')
        data = []
        for d in ds:
            turns = []
            for turn in d["conversations"]:
                turns.append({"role": "user" if turn["from"] == "human" else "assistant", "content": turn["value"]})
            data.append(turns)
    else:
        NotImplementedError
    return data

def query_chat(inp, model="gpt-3.5-turbo", temperature=1, max_tokens=256, top_p=1, n=1):
    json_data = {
      "model": model,
      "messages": [{"role": "user", "content": inp}],
      "max_tokens": max_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "n": n,
    }
    try:
        r = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data).json()
    except (requests.exceptions.JSONDecodeError, requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
        r = {'error': 1}
        print(e)
    while 'error' in r:
        eprint(r)
        time.sleep(1)
        try:
            r = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data).json()
        except Exception as e:
            print(e)
    out = r['choices'][0]['message']["content"]
    return out


if __name__ == '__main__':
    data = prepare_first_turn_user_data(sys.argv[1])
    data = [{"text": d} for d in data]
    ds = Dataset.from_list(data)
    ds = ds.train_test_split(test_size=0.3, shuffle=True)
    name = f"data/{sys.argv[1].split('/')[-1].replace('.pt', '')}_turn_first.json"
    ds["train"].to_json(name.replace('.json', '_train.json'))
    ds["test"].to_json(name.replace('.json', '_dev.json'))
    #data = prepare_complete_user_data(sys.argv[1])
    #data = [{"text": d} for d in data]
    #ds = Dataset.from_list(data)
    #name = f"data/{sys.argv[1].split('/')[-1].replace('.pt', '')}_complete.json"
    #ds.to_json(name)
    #data = prepare_response_data(sys.argv[1])
    #data = [{"text": d} for d in data]
    #ds = Dataset.from_list(data)
    #name = f"data/{sys.argv[1].split('/')[-1].replace('.pt', '')}_response.json"
    #ds.to_json(name)
    #data = convert_format(sys.argv[1])
    #torch.save(data, f"data/{sys.argv[1].split('/')[-1].replace('.pt', '')}_formatted.pt")
