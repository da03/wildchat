import sys, os

from utils import load, load_mysql, generate_hash, link_conversations, add_languages, add_detoxify, add_moderation, hash_ips, remove_wildbench, push_dataset


filename = '/root/wildchat/data/aug1_2025.pt'
print ('loading')
#load_mysql(filename)
print ('loaded')
print ('hashing')
#generate_hash(filename)
print ('hashed')
print ('linking')
#link_conversations(filename)
print ('linked')
print ('langing')
#add_languages(filename)
print ('langed')
print ('pushing')
#push_dataset(filename)
print ('pushed')
print ('rmwildbenching')
#remove_wildbench(filename)
print ('rmwildbenched')
print ('moderating')
#add_moderation(filename)
print ('moderated')
print ('detoxing')
#add_detoxify(filename)
print ('detoxified')

print ('hashing')
#hash_ips(filename)
print ('hashed')
print ('pushing before PII removal')
#push_dataset(filename, after_ip=True)
print ('pushed')
