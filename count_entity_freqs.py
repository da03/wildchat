import collections
import sys
import os
import re
import argparse
import glob
import torch
import codecs
from tqdm import tqdm
 
def main(save_name, verbose=True): 
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')] # todo
    term_freq = collections.defaultdict(collections.Counter)
    document_freq = collections.defaultdict(collections.Counter)
    files = glob.glob(f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.presidio.ner.chunk*.pt')
    # sort by numeric index after “chunk”
    def chunk_index(path):
        # this regex finds the digits between “chunk” and “.pt”
        m = re.search(r'chunk(\d+)\.pt$', path)
        return int(m.group(1)) if m else None
    files = sorted(files, key=chunk_index)
    num_conversations = 0
    total_turns = 0
    for chunk_idx, file in enumerate(files):
        print (f'loading {file}')
        d = torch.load(file, weights_only=False)
        for conversation in tqdm(d['conversation']):
            first_occurence = collections.defaultdict(set)
            num_conversations += 1
            for turn in conversation:
                text_to_anonymize = turn['content'].encode('utf-8', 'replace').decode()
                total_turns += 1
                analyzer_results = turn['analyzer_results']
                if analyzer_results is None:
                    continue
                if len(analyzer_results) > 0:
                    for analyzer_result in analyzer_results:
                        word = text_to_anonymize[analyzer_result.start:analyzer_result.end]
                        entity = analyzer_result.entity_type
                        word_l = word.strip().lower()
                        term_freq[entity][word_l] += 1
                        first_occurence[entity].add(word_l)
            for entity in first_occurence:
                for word_l in first_occurence[entity]:
                    document_freq[entity][word_l] += 1

    print (f'number of conversations: {num_conversations}')
    freqs = {'term_freq': term_freq, 'document_freq': document_freq}
    torch.save(freqs, f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.presidio.ner.entityfreq.pt')
    fouts = {}
    dirname = f'entity_freqs/wildchat_{num_conversations}'
    os.makedirs(dirname, exist_ok=True)
    if verbose:
        for entity in term_freq:
            fouts[entity] = codecs.open(os.path.join(dirname, f'{entity}.txt'), 'w', 'utf-8')
        for entity in term_freq:
            items = [(term_freq[entity][key], key) for key in term_freq[entity]]
            items = sorted(items, reverse=True)
            for count, key in items:
                fouts[entity].write(f'{key}\t{count}\n')
            fouts[entity].close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count term frequencies and document (conversation) frequencies  of entities.')
    parser.add_argument('--save_name', type=str, required=True, help='Base name of the file to process (with or without .pt extension)')

    args = parser.parse_args()

    main(args.save_name)
