from datetime import datetime
import sys
import torch
import spacy
import codecs
import os
import copy
spacy.prefer_gpu()
from tqdm import tqdm
from datasets import Dataset
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider
 
# Note: Spacy does not support Arabic pipelines
#configuration = {
#    "nlp_engine_name": "spacy",
#    "models": [
#        {"lang_code": "ENGLISH", "model_name": "en_core_web_trf"},
#        {"lang_code": "CHINESE", "model_name": "zh_core_web_trf"},
#        {"lang_code": "RUSSIAN", "model_name": "ru_core_news_lg"},
#        {"lang_code": "FRENCH", "model_name": "fr_core_news_lg"},
#        {"lang_code": "SPANISH", "model_name": "es_core_news_lg"},
#        {"lang_code": "GERMAN", "model_name": "de_core_news_lg"},
#        {"lang_code": "PORTUGUESE", "model_name": "pt_core_news_lg"},
#        {"lang_code": "ITALIAN", "model_name": "it_core_news_lg"},
#        {"lang_code": "JAPANESE", "model_name": "ja_core_news_trf"},
#        {"lang_code": "KOREAN", "model_name": "ko_core_news_lg"},
#    ],
#}
configuration = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "English", "model_name": "en_core_web_trf"},
        {"lang_code": "Chinese", "model_name": "zh_core_web_trf"},
        {"lang_code": "Russian", "model_name": "ru_core_news_lg"},
        {"lang_code": "French", "model_name": "fr_core_news_lg"},
        {"lang_code": "Spanish", "model_name": "es_core_news_lg"},
        {"lang_code": "German", "model_name": "de_core_news_lg"},
        {"lang_code": "Portuguese", "model_name": "pt_core_news_lg"},
        {"lang_code": "Italian", "model_name": "it_core_news_lg"},
        {"lang_code": "Japanese", "model_name": "ja_core_news_trf"},
        {"lang_code": "Korean", "model_name": "ko_core_news_lg"},
    ],
}

#entities = ["PHONE_NUMBER","CREDIT_CARD","CRYPTO","EMAIL_ADDRESS","IBAN_CODE","IP_ADDRESS","PERSON","MEDICAL_LICENSE"]
languages = [item['lang_code'] for item in configuration['models']]
entities = ["PHONE_NUMBER","CREDIT_CARD","CRYPTO","EMAIL_ADDRESS","IBAN_CODE","IP_ADDRESS","PERSON","MEDICAL_LICENSE", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_PASSPORT", "US_SSN", 'US_ITIN', 'UK_NHS', 'ES_NIF', 'IT_FISCAL_CODE', 'IT_DRIVER_LICENSE', 'IT_VAT_CODE', 'IT_PASSPORT', 'IT_IDENTITY_CARD', 'SG_NRIC_FIN', 'AU_TFN', 'AU_MEDICARE', 'NRP', 'LOCATION', 'DATE_TIME', 'URL']
#languages = ["SPANISH", "ENGLISH", "CHINESE", "RUSSIAN", "FRENCH", "GERMAN", "PORTUGUESE", "ITALIAN", "JAPANESE", "KOREAN"]
#ds = torch.load(sys.argv[1])

import glob

#filenames = glob.glob('latest_data_chunks_filterdate/*-deid-analyze.pt')
filenames = glob.glob('arxiv_final_data_chunks_filterdate/*-deid-analyze.pt')
#ds = []
#for filename in filenames:
#    d = torch.load(filename)
#    ds.extend(d)
filenames = sorted(list(filenames))
ds = {}
for filename in filenames:
    print (filename)
    d = torch.load(filename)
    for k in d:
        if k not in ds:
            ds[k] = []
        ds[k].extend(d[k])
#import pdb; pdb.set_trace()
# Create NLP engine based on configuration
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

# Pass the created NLP engine and supported_languages to the AnalyzerEngine
analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine, supported_languages=languages
)
anonymizer = AnonymizerEngine()

#entities = ['PERSON']
total = 0
hit = 0
thres = {'PERSON': 0.9}
import numpy as np
hit_scores = []
import collections


counts = {}
counts_dialogues = {}


for entity in entities:
    counts[entity] = collections.defaultdict(int)
    counts_dialogues[entity] = collections.defaultdict(int)
    
num_dialogues = 0
total_turns = 0
missed_turns = 0
for dialogue in tqdm(ds['conversation']):
    flags = {}
    for entity in entities:
        flags[entity] = {}
    #timestamp = dialogue[-1]['timestamp']
    cutoff_date = datetime(2024, 4, 30, 0, 0, 0)
    if not dialogue[-1]['timestamp'] <= cutoff_date:
        continue
    num_dialogues += 1


    for turn in dialogue:
        text_to_anonymize = turn['content'].encode('utf-8', 'replace').decode()
        total_turns += 1
        analyzer_results = turn['analyzer_results'] #= copy.deepcopy(analyzer_results)
        if analyzer_results is None:
            continue
        if len(analyzer_results) > 0:
            for analyzer_result in analyzer_results:
                #import pdb; pdb.set_trace()
        #        hit += 1
        #        print (hit/total, hit, total)
        #        hit_scores.append(analyzer_result.score)
        #        hit_scores_a = np.array(hit_scores)
        #        #for q in [50, 75, 90, 95, 97]:
        #        #    print (q, np.percentile(hit_scores_a, q))
                word = text_to_anonymize[analyzer_result.start:analyzer_result.end]
        #        #print (word, analyzer_result.score)
                entity = analyzer_result.entity_type
                #if entity == 'PERSON':
                #    #import pdb; pdb.set_trace()
                #    context = text_to_anonymize[max(0,analyzer_result.start-50):(analyzer_result.end+50)]
                #    #context = text_to_anonymize[analyzer_result.start-50:analyzer_result.end+50]
                #if word != word.strip():
                #    continue
                word_l = word.strip().lower()
                counts[entity][word_l] += 1
                flags[entity][word_l] = True
    for entity in entities:
        for word_l in flags[entity]:
            counts_dialogues[entity][word_l] += 1
        #        #if analyzer_result.score != 0.85:
        #        #    import pdb; pdb.set_trace()
        #        #import pdb; pdb.set_trace()
        #        if True or analyzer_result.score < thres[entity]:
        #            continue
        #        replace_by = "<ANONYMIZED>"
        #        replace_by = f'<{entity}>'
        #        result = anonymizer.anonymize(
        #            text=text_to_anonymize,
        #            analyzer_results=[analyzer_result],
        #            operators={"DEFAULT": OperatorConfig("replace", {"new_value": entity}),
        #                    "PHONE_NUMBER": OperatorConfig("mask", {"type": "mask", "masking_char" : "*", "chars_to_mask" : 12, "from_end" : True}),
        #                    "TITLE": OperatorConfig("redact", {})}
        #        )

        #        turn["content"] = result.text

#torch.save(ds, sys.argv[1].replace('.pt', '-deid-analyze.pt'))
print ('num_ia', num_dialogues)
#torch.save({'counts': counts, 'counts_dialogues': counts_dialogues}, 'all_latest_counts_analyzed_filterdate.pt')
torch.save({'counts': counts, 'counts_dialogues': counts_dialogues}, 'all_latest_counts_analyzed_filterdate_may2.pt')
#print (total_turns)
fouts = {}
os.makedirs(f'analysis_may2/all_latest_debug_{num_dialogues}', exist_ok=True)
for entity in entities:
    fouts[entity] = codecs.open(f'analysis_may2/all_latest_debug_{num_dialogues}/{entity}.txt', 'w', 'utf-8')
for entity in entities:
    items = [(counts[entity][key], key) for key in counts[entity]]
    items = sorted(items, reverse=True)
    for count, key in items:
        fouts[entity].write(f'{key}\t{count}\n')
    fouts[entity].close()
