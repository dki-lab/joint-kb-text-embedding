import sys, os
import json
import codecs
from tqdm import tqdm
import argparse
import math
from collections import Counter
import pickle
from wiki2vec.dictionary import Dictionary

def read_entity(entity_path):
    with open(entity_path) as f:
        entity2id = {}
        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    return entity2id

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--triples-file', type=str, required=True, help='path to original triples file')
    parser.add_argument('--wiki-link-file', type=str, required=True)
    parser.add_argument('--entity-file', type=str, required=True)
    parser.add_argument('--dict-file', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()

    ent_counter = Counter()
    rel_counter = Counter()

    dictionary = Dictionary.load(args.dict_file)
    wikipedia_ent = set(map(lambda x:x.title, dictionary.entities()))

    wikipedia_ent = set(map(lambda x:x.replace('_', ' '), wikipedia_ent))

    wiki_link_dict = json.load(open(args.wiki_link_file))
    entity2id = read_entity(args.entity_file)

    wikidata_ent=list(entity2id.keys())
    wikidata_ent_names=[wiki_link_dict[x] for x in wikidata_ent if x in wiki_link_dict]

    embed_vocab_set = wikipedia_ent & set(wikidata_ent_names)
    wiki_link_keys = set(wiki_link_dict.keys())
    # print(len(embed_vocab_set))

    f0 = open(args.triples_file, 'r')

    for line in f0:
        line = line.rstrip()
        h, r, t = line.split('\t')
        if h in wiki_link_keys and wiki_link_dict[h] in embed_vocab_set:
            ent_counter.update([wiki_link_dict[h]])
            rel_counter.update([r])

        if t in wiki_link_keys and wiki_link_dict[t] in embed_vocab_set:
            ent_counter.update([wiki_link_dict[t]])
            rel_counter.update([r])

    # print(len(ent_counter.keys()))
    json.dump(ent_counter, open(os.path.join(args.out_dir, 'ent_counter_names.json'), 'w'))
