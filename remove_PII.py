import sys
import torch
import spacy
import codecs
import string
import os
import copy
spacy.prefer_gpu()
from tqdm import tqdm
from datasets import Dataset
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider

def not_valid_url(s, language, context_local, full_msg):
    if 'linkedin.com/in/' in s:
        return False
    return True

def not_valid_card(s, language, context_local, full_msg):
    if 'card' not in context_local.lower():
        return True
    return False

def not_valid_email(s, language, context_local, full_msg):
    if '@example.com' in s.lower():
        return True
    if '"email":' in context_local.lower():
        return True
    if 'webmaster@' in s.lower():
        return True
    if 'login@' in s.lower():
        return True
    if '@login.com' in s.lower():
        return True
    if '@xxxxx.com' in s.lower():
        return True
    if '@xxxx.com' in s.lower():
        return True
    if '@xxx.com' in s.lower():
        return True
    if '@xx.com' in s.lower():
        return True
    if '@x.com' in s.lower():
        return True
    if 'abcdefg@' in s.lower():
        return True
    if 'info@' in s.lower():
        return True
    if 'feedback@' in s.lower():
        return True
    if 'partners@' in s.lower():
        return True
    if '@companyname.com' in s.lower():
        return True
    if 'events@' in s.lower():
        return True
    if 'orders@' in s.lower():
        return True
    if 'shipping@' in s.lower():
        return True
    if 'quality@' in s.lower():
        return True
    if 'reservations@' in s.lower():
        return True
    if 'inquiries@' in s.lower():
        return True
    if 'complaints@' in s.lower():
        return True
    if 'contact@' in s.lower():
        return True
    if 'computer@' in s.lower():
        return True
    if 'test@' in s.lower():
        return True
    if 'email@' in s.lower():
        return True
    if '@domain.com' in s.lower():
        return True
    if '@email.com' in s.lower():
        return True
    if 'your@' in s.lower()[:len('your@')]:
        return True
    if '/' in s:
        return True
    return False

def not_valid_phone(s, language, context_local, full_msg):
    if len(s) <= 8:
        return True
    if '  ' in s:
        return True
    if '192.168.' in s:
        return True
    if '127.0.' in s:
        return True
    if '255.255.' in s:
        return True
    allowed_characters = ['-', ' ', '(', ')', '+']
    for num in range(10):
        allowed_characters.append(str(num))
    for a in s:
        if a not in allowed_characters:
            return True

    required = [' m:', '\nm:', 'tel.', 'tel:', 'tel ', 'phone', 'whatsapp', 'wechat', '电话', '手机']
    flag = True
    for a in required:
        if a in context_local.lower():
            if a == 'phone':
                if ('phone":' in context_local.lower() or 'phone\':' in context_local.lower() or 'number":' in context_local.lower() or 'number\':' in context_local.lower()) and ('{' in full_msg and '}' in full_msg):
                    continue
            flag = False
    if s[0] == '+' and ('0' <= s[1] and s[1] <= '9') and len(context_local)>0 and (context_local[-1] == ' ' or context_local[-1] == '\n' or context_local[-1] == '\t'):
        flag = False
    return flag
    #return False

def not_valid_person(s, language, context_local, full_msg):
    if '\n' in s:
        return True
    if all(['a' <= c and c <= 'z' for c in s.lower()][:3]):
        language = 'English'
    for a in range(10):
        if str(a) in s:
            return True
    if language == 'English':
        escape_list = [] #['Chie Takeuchi', 'Kazumi Oshima', 'Maho Kiryu', 'Topias Kallio', 'Ayano Yamamoto', 'Ayano Osaki', 'Koyama Hiroki']
        for a in escape_list:
            if a == s:
                return True
        if ' the ' in s:
            return True
        if full_msg.lower().startswith('meow'):
            return True
        if full_msg[0] == '[' and ']' in full_msg[:20]:
            return True
        if 'write a story' in full_msg[:20]:
            return True
        if s.strip() != s:
            return True
        if len(s) <= 1:
            return True
        for punc in list(string.punctuation) + ["’", "，"]:
            if punc == '.':
                continue
            if punc == '-':
                continue
            if punc in s:
                return True
        if s[1] == '.':
            return True
        if s[1] == ' ':
            return True
        if s[0] == '.':
            return True
        if s[0] == '-':
            return True
        if s[-1] == '.':
            return True
        if s[-1] == '-':
            return True
        if '- ' in s or ' -' in s:
            return True
        if len(s.split()) < 2:
            return True
        if 'a' <= s[0] and s[0] <= 'z':
            return True
        l = ['my name is ', 'i am ', 'i\'m ', 'this is ', 'best,', 'best regard', 'best wish', 'sincerely,', 'regards', 'thanks,', 'thanks ', 'yours ', 'yours,', 'faithfully,', 'cheers,', 'take care,','dear ', 'hello ', 'hi ']
        flag = False
        for a in l:
            if a in context_local.lower():
                if a == 'my name is ' or a == 'i am ' or a == 'i\'m ' or a == 'this is ':
                    if (a).lower() in context_local.lower()[-len(a):]:
                        flag = True
                else:
                    flag = True
        if not flag:
            return True
    if language == 'Chinese':
        l = ['名字是', '叫']
        flag = False
        for a in l:
            if a in context_local:
                if a == '叫':
                    if a in context_local[-len(a):]:
                        flag = True
                else:
                    flag = True
        if not flag:
            return True
    return False
 
# Note: Spacy does not support Arabic pipelines
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

#filenames = glob.glob('arxiv_final_data_chunks_filterdate/*-deid-analyze.pt')
#filenames = sorted(list(filenames))
ds = {}
#for filename in filenames:
for i in range(100):
    filename = f'arxiv_final_data_chunks_filterdate/{i}-deid-analyze.pt'
    print (filename)
    d = torch.load(filename)
    for k in d:
        if k not in ds:
            ds[k] = []
        ds[k].extend(d[k])
    #ds.extend(d)
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
operators = {}
for entity in entities:
    replace_by = f'<PRESIDIO_ANONYMIZED_{entity}>'
    operators[entity] = OperatorConfig("replace", {"new_value": replace_by})
    
a = torch.load('all_latest_counts_analyzed_filterdate_may2.pt')
counts = a['counts']
counts_dialogues = a['counts_dialogues']
num_dialogues = 0
total_turns = 0
missed_turns = 0
turns_removed = collections.defaultdict(int)
dialogues_removed = collections.defaultdict(int)
for dialogue in tqdm(ds['conversation']):
    num_dialogues += 1
    flags = {}
    for entity in entities:
        flags[entity] = {}
    flag_removed = {}
    for entity in entities:
        flag_removed[entity] = False
    for turn in dialogue:
        text_to_anonymize = turn['content'].encode('utf-8', 'replace').decode()
        total_turns += 1
        analyzer_results = turn['analyzer_results'] #= copy.deepcopy(analyzer_results)
        if analyzer_results is None:
            continue
        analyzer_results_new = []
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
                #if entity != 'PERSON':
                #    continue
                #if turn['role'] == 'assistant':
                #    continue
                language = turn['language']
                #if entity == 'IP_ADDRESS':
                #    continue
                #if entity == 'CRYPTO':
                #    continue
                #if entity == 'MEDICAL_LICENSE':
                #    continue

                word_l = word.strip().lower()
                #if counts_dialogues[entity][word_l] >= 2:
                #    continue
                #if counts_dialogues[entity][word_l] >= 10:
                #    continue
                #if counts[entity][word_l] >= 2:
                #    continue
                context = text_to_anonymize[max(0,analyzer_result.start-50):(analyzer_result.end+50)]
                context_local = text_to_anonymize[max(0,analyzer_result.start-20):(analyzer_result.start)]
                if entity == 'PERSON':
                    if counts_dialogues[entity][word_l] >= 50:
                        continue
                    if language != 'English':
                        continue
                    if text_to_anonymize[max(0,analyzer_result.start-1)] == '"' or text_to_anonymize[max(0,analyzer_result.start-1)] == "'":
                        continue
                    if not_valid_person(word, language, context_local, turn['content']):
                        continue
                    print (entity, word, '||', language, turn['role'], context)
                elif entity == 'PHONE_NUMBER':
                    if counts_dialogues[entity][word_l] >= 10:
                        continue
                    if not_valid_phone(word, language, context_local, turn['content']):
                        continue
                    if text_to_anonymize[max(0,analyzer_result.start-1)] == '"' or text_to_anonymize[max(0,analyzer_result.start-1)] == "'":
                        continue
                    print (entity, word, '||', language, turn['role'], context)
                elif entity == 'EMAIL_ADDRESS':
                    if counts_dialogues[entity][word_l] >= 10:
                        continue
                    if not_valid_email(word, language, context_local, turn['content']):
                        continue
                    if text_to_anonymize[max(0,analyzer_result.start-1)] == '"' or text_to_anonymize[max(0,analyzer_result.start-1)] == "'":
                        continue
                    print (entity, word, '||', language, turn['role'], context)
                elif entity == 'CREDIT_CARD':
                    if counts_dialogues[entity][word_l] >= 10:
                        continue
                    if not_valid_card(word, language, context, turn['content']):
                        continue
                    print (entity, word, '||', language, turn['role'], context)
                elif entity == 'URL':
                    if counts_dialogues[entity][word_l] >= 10:
                        continue
                    if not_valid_url(word, language, context, turn['content']):
                        continue
                    print (entity, word, '||', language, turn['role'], context)
                else:
                    continue
                #import pdb; pdb.set_trace()
                turns_removed[entity] += 1
                flag_removed[entity] = True
                #counts[entity][word_l] += 1
                #flags[entity][word_l] = True
                #if analyzer_result.score != 0.85:
                #    import pdb; pdb.set_trace()
                analyzer_results_new.append(analyzer_result)
            if len(analyzer_results_new)>0:
                #import pdb; pdb.set_trace()
                result = anonymizer.anonymize(
                    text=text_to_anonymize,
                    analyzer_results=analyzer_results_new,
                    operators=operators
                )
                turn["content"] = result.text
    for entity in entities:
        if flag_removed[entity]:
            dialogues_removed[entity] += 1
    #for entity in entities:
    #    for word_l in flags[entity]:
    #        counts_dialogues[entity][word_l] += 1


torch.save(ds, 'arxiv-final-all-language-dialogue-output_1106_filtered_double_turns_may2.pt'.replace('.pt', '-deid-analyze.pt'))
#torch.save(ds, sys.argv[1].replace('.pt', '-deid-analyze.pt'))
for entity in entities:
    if turns_removed[entity] > 0:
        print (entity, 100*turns_removed[entity]/total_turns, turns_removed[entity], total_turns)
        print (entity, 100*dialogues_removed[entity]/num_dialogues, dialogues_removed[entity], num_dialogues)
#torch.save({'counts': counts, 'counts_dialogues': counts_dialogues}, 'counts_analyzed.pt')
#print (total_turns)
#fouts = {}
#os.makedirs(f'analysis/debug_{num_dialogues}', exist_ok=True)
#for entity in entities:
#    fouts[entity] = codecs.open(f'analysis/debug_{num_dialogues}/{entity}.txt', 'w', 'utf-8')
#for entity in entities:
#    items = [(counts[entity][key], key) for key in counts[entity]]
#    items = sorted(items, reverse=True)
#    for count, key in items:
#        fouts[entity].write(f'{key}\t{count}\n')
#    fouts[entity].close()
