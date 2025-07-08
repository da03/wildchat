import sys, os

from utils import load, load_mysql, generate_hash, link_conversations, add_languages, add_detoxify, add_moderation, hash_ips, remove_wildbench, push_dataset


filename = '/root/wildchat/data/jul6_2025.pt'
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
remove_wildbench(filename)
#print ('moderating')
#add_moderation(filename)
#print ('moderated')
#add_detoxify(filename)
#
#hash_ips(filename)
