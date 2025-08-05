## Data Preparation

The below code presumes access to the raw MySQL database. It will run the below sequence of functions:


* `load_mysql`: dumps the MySQL database into a list of files.
* `generate_hash`: hashes conversation content into hashes, which will be used later to link turns belonging to the same conversation together.
* `link_conversations`: links conversation turns into conversations, based on conversation content, timestamps, device fingerprints, and IPs.
* `add_languages`: detects language of each turn.
* `push_dataset`: pushes a raw version of the dataset for internal use.
* `remove_wildbench`: removes conversations reserved for WildBench.
* `add_moderation`: adds OpenAI Moderation results.
* `add_detoxify`: adds detoxify results.
* `hash_ips`: hashes IP addresses with salt to ensure nonreversibility.

```
python process_database_final.py
```

## PII Removal (not cleaned up yet)

The PII removal code assumes access to a SLURM-managed GPU cluster and uses distributed computing to process data using multiple GPUs.

First, run analyzer on every chunk (in practice, this should be run using multiple GPUs in parallel as spacy's NER is slow):

```
python run_presidio_ner.py --save_name data/aug1_2025 --chunk_idx 0
python run_presidio_ner.py --save_name data/aug1_2025 --chunk_idx 1
python run_presidio_ner.py --save_name data/aug1_2025 --chunk_idx 2
...
python run_presidio_ner.py --save_name data/aug1_2025 --chunk_idx N
```

Next, count the number of occurrences of each entity. These statistics will be later used for determining common entities (such as celebrities) that will not be removed.

```
python count_entity_freqs.py
```

Now, remove PII. The current PII removal code is developed iteratively: we check the identified named entities and add / remove rules to identify / deidentify PII. Similarly, we determine thresholds for common entities using an iterative process as well.

```
python remove_PII.py
```

Finally, release data.

```
python release_wildchat_1m.py
```
