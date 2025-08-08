import torch
import glob
import re
import argparse
import sys
import json
import hashlib
from tqdm import tqdm

from utils import push_dataset

def main(save_name):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')] # todo
    files = glob.glob(f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.presidio.ner.redacted.trufflehog.redacted.chunk*.pt')
    # sort by numeric index after “chunk”
    def chunk_index(path):
        # this regex finds the digits between “chunk” and “.pt”
        m = re.search(r'chunk(\d+)\.pt$', path)
        return int(m.group(1)) if m else None
    files = sorted(files, key=chunk_index)
    for chunk_idx, file in enumerate(files):
        print (f'loading {file}')
        d = torch.load(file, weights_only=False)
        openai_moderations = []
        detoxify_moderations = []
        conversation_hashes = []
        redacted = []
        for conversation in tqdm(d['conversation']):
            openai_moderations.append([turn['openai_moderation'] for turn in conversation])
            detoxify_moderations.append([turn['detoxify_moderation'] for turn in conversation])
            conversation_hash = []
            for turn in conversation:
                content = turn['content'].encode('utf-8', 'replace').decode()
                conversation_hash.append({'content': content, 'role': turn['role']})
                del turn['openai_moderation']
                del turn['detoxify_moderation']
                del turn['analyzer_results']
            key = hashlib.sha256(json.dumps(conversation_hash).encode('utf-8')).hexdigest()[:32]
            conversation_hashes.append(key)
            redacted.append(any([turn['redacted'] for turn in conversation]))
        d['conversation_hash'] = conversation_hashes
        d['openai_moderation'] = openai_moderations
        d['detoxify_moderation'] = detoxify_moderations
        d['redacted'] = redacted
        assert set(d.keys()) == set(['conversation_hash', 'model', 'timestamp', 'conversation', 'turn', 'language', 'openai_moderation', 'detoxify_moderation', 'toxic', 'redacted', 'state', 'country', 'hashed_ip', 'header']), d.keys()
        torch.save(d, f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.presidio.ner.redacted.trufflehog.redacted.final.chunk{chunk_idx}.pt')

    push_dataset(save_name, is_final=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Release WildChat.')
    parser.add_argument('--save_name', type=str, required=True, help='Base name of the file to process (with or without .pt extension)')

    args = parser.parse_args()

    main(args.save_name)
