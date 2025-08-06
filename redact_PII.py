import sys
import collections
import glob
import torch
import codecs
import string
import os
import copy
from tqdm import tqdm
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


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

def main(save_name):
    if save_name.endswith('.pt'):
        save_name = save_name[:-len('.pt')] # todo
    files = glob.glob(f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.presidio.ner.chunk*.pt')
    # sort by numeric index after “chunk”
    def chunk_index(path):
        # this regex finds the digits between “chunk” and “.pt”
        m = re.search(r'chunk(\d+)\.pt$', path)
        return int(m.group(1)) if m else None
    files = sorted(files, key=chunk_index)
    entities = ["PHONE_NUMBER","CREDIT_CARD","CRYPTO","EMAIL_ADDRESS","IBAN_CODE","IP_ADDRESS","PERSON","MEDICAL_LICENSE", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_PASSPORT", "US_SSN", 'US_ITIN', 'UK_NHS', 'ES_NIF', 'IT_FISCAL_CODE', 'IT_DRIVER_LICENSE', 'IT_VAT_CODE', 'IT_PASSPORT', 'IT_IDENTITY_CARD', 'SG_NRIC_FIN', 'AU_TFN', 'AU_MEDICARE', 'NRP', 'LOCATION', 'DATE_TIME', 'URL']
    operators = {}
    for entity in entities:
        replace_by = f'<PRESIDIO_ANONYMIZED_{entity}>'
        operators[entity] = OperatorConfig("replace", {"new_value": replace_by})
    anonymizer = AnonymizerEngine()
    
    num_conversations = 0
    total_turns = 0
    missed_turns = 0
    turns_removed = collections.defaultdict(int)
    conversations_removed = collections.defaultdict(int)
    document_freq = torch.load(f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.presidio.ner.entityfreq.pt')['document_freq']
    for chunk_idx, file in enumerate(files):
        print (f'loading {file}')
        d = torch.load(file, weights_only=False)
        for conversation in tqdm(d['conversation']):
            num_conversations += 1
            flag_removed = {}
            for entity in entities:
                flag_removed[entity] = False
            for turn in conversation:
                text_to_anonymize = turn['content'].encode('utf-8', 'replace').decode()
                total_turns += 1
                analyzer_results = turn['analyzer_results']
                if analyzer_results is None:
                    continue
                analyzer_results_new = []
                if len(analyzer_results) > 0:
                    for analyzer_result in analyzer_results:
                        word = text_to_anonymize[analyzer_result.start:analyzer_result.end]
                        entity = analyzer_result.entity_type
                        language = turn['language']
                        word_l = word.strip().lower()
                        context = text_to_anonymize[max(0,analyzer_result.start-50):(analyzer_result.end+50)]
                        context_local = text_to_anonymize[max(0,analyzer_result.start-20):(analyzer_result.start)]
                        if entity == 'PERSON':
                            if document_freq[entity][word_l] >= 50:
                                continue
                            if language != 'English':
                                continue
                            if text_to_anonymize[max(0,analyzer_result.start-1)] == '"' or text_to_anonymize[max(0,analyzer_result.start-1)] == "'":
                                continue
                            if not_valid_person(word, language, context_local, turn['content']):
                                continue
                            print (entity, word, '||', language, turn['role'], context)
                        elif entity == 'PHONE_NUMBER':
                            if document_freq[entity][word_l] >= 10:
                                continue
                            if not_valid_phone(word, language, context_local, turn['content']):
                                continue
                            if text_to_anonymize[max(0,analyzer_result.start-1)] == '"' or text_to_anonymize[max(0,analyzer_result.start-1)] == "'":
                                continue
                            print (entity, word, '||', language, turn['role'], context)
                        elif entity == 'EMAIL_ADDRESS':
                            if document_freq[entity][word_l] >= 10:
                                continue
                            if not_valid_email(word, language, context_local, turn['content']):
                                continue
                            if text_to_anonymize[max(0,analyzer_result.start-1)] == '"' or text_to_anonymize[max(0,analyzer_result.start-1)] == "'":
                                continue
                            print (entity, word, '||', language, turn['role'], context)
                        elif entity == 'CREDIT_CARD':
                            if document_freq[entity][word_l] >= 10:
                                continue
                            if not_valid_card(word, language, context, turn['content']):
                                continue
                            print (entity, word, '||', language, turn['role'], context)
                        elif entity == 'URL':
                            if document_freq[entity][word_l] >= 10:
                                continue
                            if not_valid_url(word, language, context, turn['content']):
                                continue
                            print (entity, word, '||', language, turn['role'], context)
                        else:
                            continue
                        turns_removed[entity] += 1
                        flag_removed[entity] = True
                        analyzer_results_new.append(analyzer_result)
                    if len(analyzer_results_new)>0:
                        result = anonymizer.anonymize(
                            text=text_to_anonymize,
                            analyzer_results=analyzer_results_new,
                            operators=operators
                        )
                        turn["content"] = result.text
            for entity in entities:
                if flag_removed[entity]:
                    conversations_removed[entity] += 1
        torch.save(ds, f'{save_name}.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.presidio.ner.redacted.chunk{chunk_idx}.pt')
    for entity in entities:
        if turns_removed[entity] > 0:
            print (entity, 100*turns_removed[entity]/total_turns, turns_removed[entity], total_turns)
            print (entity, 100*conversations_removed[entity]/num_conversations, conversations_removed[entity], num_conversations)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Redact PII based on Presidio NER results and entity frequencies.')
    parser.add_argument('--save_name', type=str, required=True, help='Base name of the file to process (with or without .pt extension)')

    args = parser.parse_args()

    main(args.save_name)
