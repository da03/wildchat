import sys, os

from utils import load, load_mysql, generate_hash, link_conversations, add_languages, add_detoxify, add_moderation, hash_ips, remove_wildbench


#baseline_file = 'output.baseline.csv'
#load(baseline_file)


#csv_file = 'output.head.csv'
#load_csv(csv_file)

#filename = '/root/database_backup_mar16.sql'
#filename = '/root/dumps_include9/wildchat.mar16.2024.include.9.pt'
filename = '/root/wildchat/data/may25_2025.pt'
print ('loading')
#load_mysql(filename)
print ('loaded')
print ('hashing')
generate_hash(filename)
print ('hashed')
#print ('linking')
#link_conversations(filename)
#print ('linked')
#print ('langing')
#add_languages(filename)
#print ('langed')
#remove_wildbench(filename)
#print ('moderating')
#add_moderation(filename)
#print ('moderated')
#add_detoxify(filename)

#hash_ips(filename)
