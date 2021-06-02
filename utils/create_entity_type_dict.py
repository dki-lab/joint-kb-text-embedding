import codecs, json, sys, os
import logging
import argparse
import gzip
from tqdm import tqdm
from collections import Counter

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikidata-triples-file', type=str, required=True, help='path to wikidata json dir.')
    parser.add_argument('--out-dir', type=str, required=True, help='path to output dir. of dicts')
    args = parser.parse_args()

    f1 = open(args.wikidata_triples_file, 'r')

    entity_type_dict = {}
    entity_child_dict = {}

    for line in f1:
        h, r, t = line.rstrip().split('\t')
        if r == 'P31' or r == 'P279' or r == 'P361':
            if h in entity_type_dict:
                entity_type_dict[h].append(t)
            else:
                entity_type_dict[h] = [t]

            if t in entity_child_dict:
                entity_child_dict[t].append(h)
            else:
                entity_child_dict[t] = [h]

    f1.close()
    json.dump(entity_type_dict, open(os.path.join(args.out_dir, 'entity_type_dict.json'), 'w'))
    json.dump(entity_child_dict, open(os.path.join(args.out_dir, 'entity_child_dict.json'), 'w'))
