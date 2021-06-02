import codecs, json, sys, os
import logging
import argparse
import gzip
from tqdm import tqdm
from collections import Counter
import pickle
# @profile
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wikidata-triples-file', type=str, required=True, help='path to wikidata json dir.')
    parser.add_argument('--entity-type-dict-file', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True, help='path to output dir. of dicts')
    args = parser.parse_args()

    rel_type_dict = {}
    entity_type_dict = json.load(open(args.entity_type_dict_file, 'r'))

    f1 = open(args.wikidata_triples_file, 'r')

    for line in f1:
        h, r, t = line.rstrip().split('\t')

        if r in rel_type_dict:
            if h in entity_type_dict:
                if 'head_types' in rel_type_dict[r]:
                    rel_type_dict[r]['head_types'].update(entity_type_dict[h])
                else:
                    rel_type_dict[r]['head_types'] = set(entity_type_dict[h])

            if t in entity_type_dict:
                if 'tail_types' in rel_type_dict[r]:
                    rel_type_dict[r]['tail_types'].update(entity_type_dict[t])
                else:
                    rel_type_dict[r]['tail_types'] = set(entity_type_dict[t])
        else:
            rel_type_dict[r] = {}
            if h in entity_type_dict:
                rel_type_dict[r]['head_types'] = set(entity_type_dict[h])

            if t in entity_type_dict:
                rel_type_dict[r]['tail_types'] = set(entity_type_dict[t])

    f1.close()
    pickle.dump(rel_type_dict, open(os.path.join(args.out_dir, 'rel_type_dict.pickle'), 'wb'))


if __name__=='__main__':
    main()