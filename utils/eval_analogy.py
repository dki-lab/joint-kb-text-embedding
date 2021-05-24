import argparse
import torch
import torch.nn.functional as F
import gensim
from gensim import utils
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import random
from collections import Counter
import pickle
import json
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from walker import WalkerRandomSampling

'''
Usage: PYTHONHASHSEED=12345 python eval_analogy.py --sg-ckpt-file ~/data/projection-human-expt/reg_coeff_10/FB15k_TransE_l2_sg_5_ent.bin --analogy-file wikipedia_analogy_toy.txt --entity-counter-file /home/pahuja.9/data/wikidata_proc_json/wikipedia_entity_counter.json --wiki-link-file ~/data/wikipedia_links.json --entity-file ~/data/wikidata_triples_human/entities.tsv
'''
def normalize(x):
    return x/np.linalg.norm(x)

def random_sample_uniform_neg(embed_vocab_list, exclude_list, num_neg):
    out = set()
    while len(out)<num_neg:
        o = random.choice(embed_vocab_list)
        if o not in exclude_list:
            out.add(o)
    return out

def random_sample_degree_neg(embed_vocab_list, exclude_list, num_neg, wrand, ent_counter):
    out = set()
    # source_set = embed_vocab_list - set([t])
    # out = np.random.choice(source_set, weights=prob, replace=False)
    while len(out)<num_neg:
        o = wrand.random(1)[0]
        # print(o)
        # print(o[0])
        # print('count = {}'.format(ent_counter[o.replace('_',' ')]))
        if o not in exclude_list:
            out.add(o)
    return out

def read_entity(entity_path):
    with open(entity_path) as f:
        entity2id = {}
        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    return entity2id

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sg-ckpt-file', type=str, required=True)
    parser.add_argument('--analogy-file', type=str, required=True)
    parser.add_argument('--entity-counter-file', type=str, required=True)
    parser.add_argument('--wiki-link-file', type=str, required=True)
    parser.add_argument('--entity-file', type=str, required=True)
    parser.add_argument('--num-neg', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--uniform-sampling', action='store_true')
    parser.add_argument('--degree-sampling', action='store_true')
    # parser.add_argument('--n', type=int, default=10)

    args = parser.parse_args()

    if not args.uniform_sampling and not args.degree_sampling:
        args.uniform_sampling = True

    ent_counter = json.load(open(args.entity_counter_file))

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cpu')

    embed = KeyedVectors.load_word2vec_format(args.sg_ckpt_file, binary=True)
    wiki_link_dict = json.load(open(args.wiki_link_file))
    entity2id = read_entity(args.entity_file)
    wikidata_ent=list(entity2id.keys())
    wikidata_ent_names=[wiki_link_dict[x] for x in wikidata_ent if x in wiki_link_dict]

    wikipedia_ent = set(map(lambda x:x.replace('_', ' '), embed.vocab.keys()))

    print('len(embed.vocab) = {}'.format(len(embed.vocab.keys())))
    print('len(wikipedia_ent) = {}'.format(len(wikipedia_ent)))
    print('len(wikidata_ent_names) = {}'.format(len(wikidata_ent_names)))
    # print(embed)
    embed_vocab_list = list(wikipedia_ent & set(wikidata_ent_names))
    print('len(embed_vocab_list) = {}'.format(len(embed_vocab_list)))

    embed_vocab_freq = [ent_counter[x] for x in tqdm(embed_vocab_list)]
    
    # print(embed_vocab_idx_tbl)
    embed_vocab_list = [x.replace(' ', '_') for x in embed_vocab_list]

    # print(embed_vocab_freq)
    wrand = WalkerRandomSampling(embed_vocab_freq, embed_vocab_list)
    # freq_sum = sum(embed_vocab_freq)
    # embed_vocab_prob = [x*1.0/freq_sum for x in embed_vocab_freq]

    num_lines = sum([1 for line in open(args.analogy_file, 'r')])
    f1 = open(args.analogy_file, 'r')

    rank_sum = 0
    mrr_sum = 0
    n_ex = 0
    hits_10 = 0
    hits_1 = 0

    for line in tqdm(f1, total=num_lines):
        line = line.rstrip()
        if line.startswith(': '):
            continue

        h1, t1, h2, t2 = line.split(' ')

        # print('h1 = {}, t1 = {}, h2 = {}, t2 = {}'.format(h1, t1, h2, t2))

        # print(embed.get_vector(h1))
        # print(embed.word_vec)
        embed_h1 = embed.get_vector(h1)
        embed_h2 = embed.get_vector(h2)
        embed_t1 = embed.get_vector(t1)
        embed_t2 = embed.get_vector(t2)

        embed_h1 = normalize(embed_h1)
        embed_t1 = normalize(embed_t1)
        embed_h2 = normalize(embed_h2)
        embed_t2 = normalize(embed_t2)

        exp_embed_t2 = (embed_t1 - embed_h1 + embed_h2) # [embed_dim]
        
        # negatives = random.sample(set(embed.vocab.keys()) - set([t2]), args.num_neg)
        negatives = []
        if args.uniform_sampling:
            negatives_uniform = random_sample_uniform_neg(embed_vocab_list, [h1, t1, h2, t2], args.num_neg)
            negatives.extend(negatives_uniform)

        if args.degree_sampling:
            negatives_degree = random_sample_degree_neg(embed_vocab_list, [h1, t1, h2, t2], args.num_neg, wrand, ent_counter)
            negatives.extend(negatives_degree)

        # print('negatives = {}'.format(negatives))

        negatives_embed = np.vstack([normalize(embed.get_vector(x)) for x in negatives]) # [num_neg, embed_dim]

        exp_embed_t2_repeat = np.tile(exp_embed_t2, (args.num_neg, 1)) # [num_neg, embed_dim]
        embed_gold_prod = np.tile((embed_t2 * exp_embed_t2).sum(), (args.num_neg)) # [num_neg]

        # print((embed_t2 * exp_embed_t2).sum())

        prod = torch.tensor((exp_embed_t2_repeat * negatives_embed).sum(axis=1)) # [num_neg]
        embed_gold_prod = torch.tensor(embed_gold_prod)

        # print('prod = {}'.format(prod.size()))
        # print('embed_gold_prod = {}'.format(embed_gold_prod.size()))

        rank = 1+torch.nonzero(F.relu(prod - embed_gold_prod), as_tuple=False).size(0)
        # print(prod - embed_gold_prod)
        # print('rank = {}'.format(rank))
        # print(prod)
        # print((((prod - embed_gold_prod) > (embed_t2 * exp_embed_t2).sum())==1).to(device, dtype=torch.int32).sum())
        rank_sum += rank
        mrr_sum += 1.0/rank
        n_ex += 1

        if rank <=10:
            hits_10 += 1

        if rank ==1:
            hits_1 += 1

    mr = rank_sum/n_ex
    mrr = mrr_sum/n_ex
    hits_10 = hits_10*100.0/n_ex
    hits_1 = hits_1*100.0/n_ex

    print(args.analogy_file)
    print(f'MR = {mr:0.2f}')
    print(f'MRR = {mrr:0.4f}')
    print(f'hits@1 = {hits_1:0.2f}%')
    print(f'hits@10 = {hits_10:0.2f}%')
    print()







