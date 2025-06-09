import sys, os, glob, torch

output_dir = 'arxiv_final_data_chunks_filterdate'
os.makedirs(output_dir, exist_ok=True)
#ds = torch.load(sys.argv[1])
ds = torch.load('final.cacheddict.withlang.rmwildbench.moderations.detoxify.ip.pt')

num_partitions = 100

def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    a = []
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        a.append(l[si:si+(d+1 if i < r else d)])
    return a

def chunks_dict(d, n):
    a = {}
    for k in d:
        chunks_k = chunks(d[k], n)
        a[k] = chunks_k

    results = []
    for i in range(n):
        results.append({k: a[k][i] for k in d})
    return results

a = chunks_dict(ds, num_partitions)
#import pdb; pdb.set_trace()
count = 0
for i in range(num_partitions):
    b = a[i]
    c = []
    #for items in b['conversation']:
    #    flag = True
    #    for item in items:
    #        content = item['content'].lower()
    #        if 'timestamp' in item:
    #            #import pdb; pdb.set_trace()
    #            from datetime import datetime
    #            datetime_object = item['timestamp']
    #            #datetime_object = datetime.strptime(item['timestamp'], '%Y-%m-%d %H:%M:%S')
    #            anc_datetime_object = datetime.strptime('2023-04-09 00:00:00', '%Y-%m-%d %H:%M:%S')
    #            if datetime_object <= anc_datetime_object:
    #                flag = False
    #                print (item['timestamp'])
    #        if 'yuntian' in content:
    #            content = content.replace('yuntian could', '')
    #            content = content.replace('yuntian-deng', '')
    #            content = content.replace('yuntian d', '')
    #            if 'yuntian' in content:
    #                #import pdb; pdb.set_trace()
    #                flag = False
    #                #count += 1
    #                print (content)
    #                #print (count, content)
    #                break
    #    if flag:
    #        pass
    #        #c.append(items)
    #    else:
    #        count += 1
    #        print ('shouldn\'t happen', count)
    torch.save(b, os.path.join(output_dir, f'{i}.pt'))
