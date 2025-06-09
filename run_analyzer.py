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



import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

 
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

languages = [item['lang_code'] for item in configuration['models']]
entities = ["PHONE_NUMBER","CREDIT_CARD","CRYPTO","EMAIL_ADDRESS","IBAN_CODE","IP_ADDRESS","PERSON","MEDICAL_LICENSE", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_PASSPORT", "US_SSN", 'US_ITIN', 'UK_NHS', 'ES_NIF', 'IT_FISCAL_CODE', 'IT_DRIVER_LICENSE', 'IT_VAT_CODE', 'IT_PASSPORT', 'IT_IDENTITY_CARD', 'SG_NRIC_FIN', 'AU_TFN', 'AU_MEDICARE', 'NRP', 'LOCATION', 'DATE_TIME', 'URL']
ds = torch.load(sys.argv[1])
# Create NLP engine based on configuration
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

# Pass the created NLP engine and supported_languages to the AnalyzerEngine
analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine, supported_languages=languages
)
anonymizer = AnonymizerEngine()

def __utf8len(s:str):
    return len(s.encode('utf-8'))

# splits not after x bytes but ensures that max x bytes are used without destroying the final character 
def __chunk_text_on_bytes(text: str, max_chunk_size: int = 1_000_000):
    factor = len(text) / __utf8len(text)
    increase_by = int(max(min(max_chunk_size*.1,10),1))
    initial_size_guess = int(max(max_chunk_size * factor - 10,1))
    print ('initial_size_guess', initial_size_guess)
    final_list = []
    remaining = text
    while len(remaining):
        part = remaining[:initial_size_guess]
        if __utf8len(part) > max_chunk_size:
            initial_size_guess = int(max(initial_size_guess - min(max_chunk_size *.001,10),1) )
            print ('initial_size_guess', initial_size_guess)
            continue
        cut_after = initial_size_guess
        while __utf8len(part) < max_chunk_size and part != remaining:
            cut_after = min(len(remaining), cut_after+increase_by)
            part = remaining[:cut_after]
            
        if __utf8len(part) > max_chunk_size:
            cut_after-=increase_by
        final_list.append(remaining[:cut_after])
        remaining = remaining[cut_after:]

    return final_list


#entities = ['PERSON']
total = 0
hit = 0
thres = {'PERSON': 0.9}
import numpy as np
hit_scores = []
#import pdb; pdb.set_trace()
num_dialogues = 0
total_turns = 0
missed_turns = 0
for dialogue in tqdm(ds['conversation']):
    for turn in dialogue:
        text_to_anonymize = turn['content'].encode('utf-8', 'replace').decode()
        print (len(text_to_anonymize))
        total_turns += 1
        if turn["language"] not in languages:
            missed_turns += 1
            print (missed_turns / total_turns, missed_turns, total_turns)
            analyzer_results = None
        else:
            try:
                with time_limit(60):
                    analyzer_results = analyzer.analyze(text=text_to_anonymize, entities=entities, language=turn["language"])
            except Exception as e:
                #import pdb; pdb.set_trace()
                err_msg = f'{e}'
                print (err_msg)

                if '49149' in err_msg:
                    print ('case A')
                    chunks = __chunk_text_on_bytes(text_to_anonymize, 49000)
                    offset = 0
                    analyzer_results = []
                    for text in chunks:
                        analyzer_results_i = analyzer.analyze(text=text, entities=entities, language=turn["language"])
                        for analyzer_result in analyzer_results_i:
                            analyzer_result.start = analyzer_result.start + offset
                            analyzer_result.end = analyzer_result.end + offset
                            analyzer_results.append(analyzer_result)
                        offset += len(text)
                elif type(e) == TimeoutException:
                    #import pdb; pdb.set_trace()
                    print ('case B')
                    #chunks = __chunk_text_on_bytes(text_to_anonymize, 49000)
                    chunks = __chunk_text_on_bytes(text_to_anonymize, 4096)
                    offset = 0
                    analyzer_results = []
                    for text in chunks:
                        print ('*')
                        analyzer_results_i = analyzer.analyze(text=text, entities=entities, language=turn["language"])
                        for analyzer_result in analyzer_results_i:
                            analyzer_result.start = analyzer_result.start + offset
                            analyzer_result.end = analyzer_result.end + offset
                            analyzer_results.append(analyzer_result)
                        offset += len(text)
                else:
                    print ('case C')
                    raise e

        turn['analyzer_results'] = copy.deepcopy(analyzer_results)
        total += 1

torch.save(ds, sys.argv[1].replace('.pt', '-deid-analyze.pt'))
