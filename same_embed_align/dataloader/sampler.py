# -*- coding: utf-8 -*-
#
# sampler.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import numpy as np
import scipy as sp
import dgl.backend as F
import dgl
import os
import sys
import pickle
import time
import torch

from dgl.base import NID, EID
from tqdm import tqdm

def SoftRelationPartition(edges, n, threshold=0.05):
    """This partitions a list of edges to n partitions according to their
    relation types. For any relation with number of edges larger than the
    threshold, its edges will be evenly distributed into all partitions.
    For any relation with number of edges smaller than the threshold, its
    edges will be put into one single partition.

    Algo:
    For r in relations:
        if r.size() > threadold
            Evenly divide edges of r into n parts and put into each relation.
        else
            Find partition with fewest edges, and put edges of r into 
            this partition.

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        Number of partitions
    threshold : float
        The threshold of whether a relation is LARGE or SMALL
        Default: 5%

    Returns
    -------
    List of np.array
        Edges of each partition
    List of np.array
        Edge types of each partition
    bool
        Whether there exists some relations belongs to multiple partitions
    """
    heads, rels, tails = edges
    print('relation partition {} edges into {} parts'.format(len(heads), n))
    uniq, cnts = np.unique(rels, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]
    assert cnts[0] > cnts[-1]
    edge_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_dict = {}
    rel_parts = []
    cross_rel_part = []
    for _ in range(n):
        rel_parts.append([])

    large_threshold = int(len(rels) * threshold)
    capacity_per_partition = int(len(rels) / n)
    # ensure any relation larger than the partition capacity will be split
    large_threshold = capacity_per_partition if capacity_per_partition < large_threshold \
                      else large_threshold
    num_cross_part = 0
    for i in range(len(cnts)):
        cnt = cnts[i]
        r = uniq[i]
        r_parts = []
        if cnt > large_threshold:
            avg_part_cnt = (cnt // n) + 1
            num_cross_part += 1
            for j in range(n):
                part_cnt = avg_part_cnt if cnt > avg_part_cnt else cnt
                r_parts.append([j, part_cnt])
                rel_parts[j].append(r)
                edge_cnts[j] += part_cnt
                rel_cnts[j] += 1
                cnt -= part_cnt
            cross_rel_part.append(r)
        else:
            idx = np.argmin(edge_cnts)
            r_parts.append([idx, cnt])
            rel_parts[idx].append(r)
            edge_cnts[idx] += cnt
            rel_cnts[idx] += 1
        rel_dict[r] = r_parts

    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} relations'.format(i, edge_cnt, rel_cnts[i]))
    print('{}/{} duplicated relation across partitions'.format(num_cross_part, len(cnts)))

    parts = []
    for i in range(n):
        parts.append([])
        rel_parts[i] = np.array(rel_parts[i])

    for i, r in enumerate(rels):
        r_part = rel_dict[r][0]
        part_idx = r_part[0]
        cnt = r_part[1]
        parts[part_idx].append(i)
        cnt -= 1
        if cnt == 0:
            rel_dict[r].pop(0)
        else:
            rel_dict[r][0][1] = cnt

    for i, part in enumerate(parts):
        parts[i] = np.array(part, dtype=np.int64)
    shuffle_idx = np.concatenate(parts)
    heads[:] = heads[shuffle_idx]
    rels[:] = rels[shuffle_idx]
    tails[:] = tails[shuffle_idx]

    off = 0
    for i, part in enumerate(parts):
        parts[i] = np.arange(off, off + len(part))
        off += len(part)
    cross_rel_part = np.array(cross_rel_part)

    return parts, rel_parts, num_cross_part > 0, cross_rel_part

def BalancedRelationPartition(edges, n):
    """This partitions a list of edges based on relations to make sure
    each partition has roughly the same number of edges and relations.
    Algo:
    For r in relations:
      Find partition with fewest edges
      if r.size() > num_of empty_slot
         put edges of r into this partition to fill the partition,
         find next partition with fewest edges to put r in.
      else
         put edges of r into this partition.

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        number of partitions

    Returns
    -------
    List of np.array
        Edges of each partition
    List of np.array
        Edge types of each partition
    bool
        Whether there exists some relations belongs to multiple partitions
    """
    heads, rels, tails = edges
    print('relation partition {} edges into {} parts'.format(len(heads), n))
    uniq, cnts = np.unique(rels, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]
    assert cnts[0] > cnts[-1]
    edge_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_dict = {}
    rel_parts = []
    for _ in range(n):
        rel_parts.append([])

    max_edges = (len(rels) // n) + 1
    num_cross_part = 0
    for i in range(len(cnts)):
        cnt = cnts[i]
        r = uniq[i]
        r_parts = []

        while cnt > 0:
            idx = np.argmin(edge_cnts)
            if edge_cnts[idx] + cnt <= max_edges:
                r_parts.append([idx, cnt])
                rel_parts[idx].append(r)
                edge_cnts[idx] += cnt
                rel_cnts[idx] += 1
                cnt = 0
            else:
                cur_cnt = max_edges - edge_cnts[idx]
                r_parts.append([idx, cur_cnt])
                rel_parts[idx].append(r)
                edge_cnts[idx] += cur_cnt
                rel_cnts[idx] += 1
                num_cross_part += 1
                cnt -= cur_cnt
        rel_dict[r] = r_parts

    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} relations'.format(i, edge_cnt, rel_cnts[i]))
    print('{}/{} duplicated relation across partitions'.format(num_cross_part, len(cnts)))

    parts = []
    for i in range(n):
        parts.append([])
        rel_parts[i] = np.array(rel_parts[i])

    for i, r in enumerate(rels):
        r_part = rel_dict[r][0]
        part_idx = r_part[0]
        cnt = r_part[1]
        parts[part_idx].append(i)
        cnt -= 1
        if cnt == 0:
            rel_dict[r].pop(0)
        else:
            rel_dict[r][0][1] = cnt

    for i, part in enumerate(parts):
        parts[i] = np.array(part, dtype=np.int64)
    shuffle_idx = np.concatenate(parts)
    heads[:] = heads[shuffle_idx]
    rels[:] = rels[shuffle_idx]
    tails[:] = tails[shuffle_idx]

    off = 0
    for i, part in enumerate(parts):
        parts[i] = np.arange(off, off + len(part))
        off += len(part)

    return parts, rel_parts, num_cross_part > 0

def RandomPartition(edges, n):
    """This partitions a list of edges randomly across n partitions

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        number of partitions

    Returns
    -------
    List of np.array
        Edges of each partition
    """
    heads, rels, tails = edges
    print('random partition {} edges into {} parts'.format(len(heads), n))
    idx = np.random.permutation(len(heads))
    heads[:] = heads[idx]
    rels[:] = rels[idx]
    tails[:] = tails[idx]

    part_size = int(math.ceil(len(idx) / n))
    parts = []
    for i in range(n):
        start = part_size * i
        end = min(part_size * (i + 1), len(idx))
        parts.append(idx[start:end])
        print('part {} has {} edges'.format(i, len(parts[-1])))
    return parts

def ConstructGraph(edges, n_entities, args):
    """Construct Graph for training

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list
    n_entities : int
        number of entities
    args :
        Global configs.
    """
    src, etype_id, dst = edges
    coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])
    g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
    g.edata['tid'] = F.tensor(etype_id, F.int64)
    return g

class TrainDataset(object):
    """Dataset for training

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    ranks:
        Number of partitions.
    """
    def __init__(self, dataset, args, ranks=64):
        triples = dataset.train
        num_train = len(triples[0])
        print('|Train|:', num_train)

        if ranks > 1 and args.rel_part:
            self.edge_parts, self.rel_parts, self.cross_part, self.cross_rels = \
            SoftRelationPartition(triples, ranks)
        elif ranks > 1:
            self.edge_parts = RandomPartition(triples, ranks)
            self.cross_part = True
        else:
            self.edge_parts = [np.arange(num_train)]
            self.rel_parts = [np.arange(dataset.n_relations)]
            self.cross_part = False

        self.g = ConstructGraph(triples, dataset.n_entities, args)

    def create_sampler(self, batch_size, neg_sample_size=2, neg_chunk_size=None, mode='head', num_workers=32,
                       shuffle=True, exclude_positive=False, rank=0):
        """Create sampler for training

        Parameters
        ----------
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        shuffle : bool
            If True, shuffle the seed edges.
            If False, do not shuffle the seed edges.
            Default: False
        exclude_positive : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
            Default: False
        rank : int
            Which partition to sample.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        assert batch_size % neg_sample_size == 0, 'batch_size should be divisible by B'
        return EdgeSampler(self.g,
                           seed_edges=F.tensor(self.edge_parts[rank]),
                           batch_size=batch_size,
                           neg_sample_size=int(neg_sample_size/neg_chunk_size),
                           chunk_size=neg_chunk_size,
                           negative_mode=mode,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           exclude_positive=exclude_positive,
                           return_false_neg=False)

class ChunkNegEdgeSubgraph(dgl.DGLGraph):
    """Wrapper for negative graph

        Parameters
        ----------
        neg_g : DGLGraph
            Graph holding negative edges.
        num_chunks : int
            Number of chunks in sampled graph.
        chunk_size : int
            Info of chunk_size.
        neg_sample_size : int
            Info of neg_sample_size.
        neg_head : bool
            If True, negative_mode is 'head'
            If False, negative_mode is 'tail'
    """
    def __init__(self, subg, num_chunks, chunk_size,
                 neg_sample_size, neg_head):
        super(ChunkNegEdgeSubgraph, self).__init__(graph_data=subg.sgi.graph,
                                                   readonly=True,
                                                   parent=subg._parent)
        self.ndata[NID] = subg.sgi.induced_nodes.tousertensor()
        self.edata[EID] = subg.sgi.induced_edges.tousertensor()
        self.subg = subg
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.neg_sample_size = neg_sample_size
        self.neg_head = neg_head

    @property
    def head_nid(self):
        return self.subg.head_nid

    @property
    def tail_nid(self):
        return self.subg.tail_nid

def create_neg_subgraph(pos_g, neg_g, chunk_size, neg_sample_size, is_chunked,
                        neg_head, num_nodes):
    """KG models need to know the number of chunks, the chunk size and negative sample size
    of a negative subgraph to perform the computation more efficiently.
    This function tries to infer all of these information of the negative subgraph
    and create a wrapper class that contains all of the information.

    Parameters
    ----------
    pos_g : DGLGraph
        Graph holding positive edges.
    neg_g : DGLGraph
        Graph holding negative edges.
    chunk_size : int
        Chunk size of negative subgrap.
    neg_sample_size : int
        Negative sample size of negative subgrap.
    is_chunked : bool
        If True, the sampled batch is chunked.
    neg_head : bool
        If True, negative_mode is 'head'
        If False, negative_mode is 'tail'
    num_nodes: int
        Total number of nodes in the whole graph.

    Returns
    -------
    ChunkNegEdgeSubgraph
        Negative graph wrapper
    """
    assert neg_g.number_of_edges() % pos_g.number_of_edges() == 0
    # We use all nodes to create negative edges. Regardless of the sampling algorithm,
    # we can always view the subgraph with one chunk.
    if (neg_head and len(neg_g.head_nid) == num_nodes) \
            or (not neg_head and len(neg_g.tail_nid) == num_nodes):
        num_chunks = 1
        chunk_size = pos_g.number_of_edges()
    elif is_chunked:
        # This is probably for evaluation.
        if pos_g.number_of_edges() < chunk_size \
                and neg_g.number_of_edges() % neg_sample_size == 0:
            num_chunks = 1
            chunk_size = pos_g.number_of_edges()
        # This is probably the last batch in the training. Let's ignore it.
        elif pos_g.number_of_edges() % chunk_size > 0:
            return None
        else:
            num_chunks = int(pos_g.number_of_edges() / chunk_size)
        assert num_chunks * chunk_size == pos_g.number_of_edges()
    else:
        num_chunks = pos_g.number_of_edges()
        chunk_size = 1
    return ChunkNegEdgeSubgraph(neg_g, num_chunks, chunk_size,
                                neg_sample_size, neg_head)

class EvalSampler(object):
    """Sampler for validation and testing

    Parameters
    ----------
    g : DGLGraph
        Graph containing KG graph
    edges : tensor
        Seed edges
    batch_size : int
        Batch size of each mini batch.
    neg_sample_size : int
        How many negative edges sampled for each node.
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    mode : str
        Sampling mode.
    number_workers: int
        Number of workers used in parallel for this sampler
    filter_false_neg : bool
        If True, exlucde true positive edges in sampled negative edges
        If False, return all sampled negative edges even there are positive edges
        Default: True
    """
    def __init__(self, g, edges, batch_size, neg_sample_size, neg_chunk_size, mode, num_workers=32,
                 filter_false_neg=True):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        self.sampler = EdgeSampler(g,
                                   batch_size=batch_size,
                                   seed_edges=edges,
                                   neg_sample_size=neg_sample_size,
                                   chunk_size=neg_chunk_size,
                                   negative_mode=mode,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   exclude_positive=False,
                                   relations=g.edata['tid'],
                                   return_false_neg=filter_false_neg)
        self.sampler_iter = iter(self.sampler)
        self.mode = mode
        self.neg_head = 'head' in mode
        self.g = g
        self.filter_false_neg = filter_false_neg
        self.neg_chunk_size = neg_chunk_size
        self.neg_sample_size = neg_sample_size

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch

        Returns
        -------
        DGLGraph
            Sampled positive graph
        ChunkNegEdgeSubgraph
            Negative graph wrapper
        """
        while True:
            pos_g, neg_g = next(self.sampler_iter)
            if self.filter_false_neg:
                neg_positive = neg_g.edata['false_neg']
            neg_g = create_neg_subgraph(pos_g, neg_g, 
                                        self.neg_chunk_size, 
                                        self.neg_sample_size, 
                                        'chunk' in self.mode, 
                                        self.neg_head, 
                                        self.g.number_of_nodes())
            if neg_g is not None:
                break

        pos_g.ndata['id'] = pos_g.parent_nid
        neg_g.ndata['id'] = neg_g.parent_nid
        pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
        if self.filter_false_neg:
            neg_g.edata['bias'] = F.astype(-neg_positive, F.float32)
        return pos_g, neg_g

    def reset(self):
        """Reset the sampler
        """
        self.sampler_iter = iter(self.sampler)
        return self

class EvalSamplerWithConstrain(object):
    """Sampler for validation and testing

    Parameters
    ----------
    g : DGLGraph
        Graph containing KG graph
    edges : tensor
        Seed edges
    batch_size : int
        Batch size of each mini batch.
    neg_sample_size : int
        How many negative edges sampled for each node.
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    mode : str
        Sampling mode.
    number_workers: int
        Number of workers used in parallel for this sampler
    filter_false_neg : bool
        If True, exlucde true positive edges in sampled negative edges
        If False, return all sampled negative edges even there are positive edges
        Default: True
    """
    def __init__(self, g, edges, probs, batch_size, neg_sample_size, neg_chunk_size, mode, num_workers=32,
                 filter_false_neg=True):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        # print('flag 1')
        print('edges = {}'.format(edges.size()))
        # x=torch.ones(edges.shape[0])
        # print(type(x))
        self.sampler = EdgeSampler(g,
                                   batch_size=batch_size,
                                   seed_edges=edges,
                                   neg_sample_size=neg_sample_size,
                                   chunk_size=neg_chunk_size,
                                   negative_mode=mode,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   exclude_positive=False,
                                   relations=g.edata['tid'],
                                   return_false_neg=filter_false_neg,
                                   # edge_weight=torch.ones(edges.size(0)),
                                   edge_weight=torch.ones(g.number_of_edges()),
                                   node_weight = probs)
        print('sampler created')
        self.sampler_iter = iter(self.sampler)
        self.mode = mode
        self.neg_head = 'head' in mode
        self.g = g
        self.filter_false_neg = filter_false_neg
        self.neg_chunk_size = neg_chunk_size
        self.neg_sample_size = neg_sample_size
        self.probs = probs

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch

        Returns
        -------
        DGLGraph
            Sampled positive graph
        ChunkNegEdgeSubgraph
            Negative graph wrapper
        """
        print('flag 1')
        while True:
            print('flag 11')
            pos_g, neg_g = next(self.sampler_iter)
            if self.filter_false_neg:
                neg_positive = neg_g.edata['false_neg']
            neg_g = create_neg_subgraph(pos_g, neg_g, 
                                        self.neg_chunk_size, 
                                        self.neg_sample_size, 
                                        'chunk' in self.mode, 
                                        self.neg_head, 
                                        self.g.number_of_nodes())
            if neg_g is not None:
                break

        print('flag 2')

        pos_g.ndata['id'] = pos_g.parent_nid
        neg_g.ndata['id'] = neg_g.parent_nid
        pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
        if self.filter_false_neg:
            neg_g.edata['bias'] = F.astype(-neg_positive, F.float32)

        return pos_g, neg_g

    def reset(self):
        """Reset the sampler
        """
        self.sampler_iter = iter(self.sampler)
        return self

class EvalDataset(object):
    """Dataset for validation or testing

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    """
    def __init__(self, dataset, args):
        src = [dataset.train[0]]
        etype_id = [dataset.train[1]]
        dst = [dataset.train[2]]
        self.num_train = len(dataset.train[0])
        if dataset.valid is not None:
            src.append(dataset.valid[0])
            etype_id.append(dataset.valid[1])
            dst.append(dataset.valid[2])
            self.num_valid = len(dataset.valid[0])
        else:
            self.num_valid = 0
        if dataset.test is not None:
            src.append(dataset.test[0])
            etype_id.append(dataset.test[1])
            dst.append(dataset.test[2])
            self.num_test = len(dataset.test[0])
        else:
            self.num_test = 0
        assert len(src) > 1, "we need to have at least validation set or test set."
        src = np.concatenate(src)
        etype_id = np.concatenate(etype_id)
        dst = np.concatenate(dst)

        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
                                    shape=[dataset.n_entities, dataset.n_entities])
        g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
        g.edata['tid'] = F.tensor(etype_id, F.int64)
        self.g = g

        if args.eval_percent < 1:
            self.valid = np.random.randint(0, self.num_valid,
                    size=(int(self.num_valid * args.eval_percent),)) + self.num_train
        else:
            self.valid = np.arange(self.num_train, self.num_train + self.num_valid)
        print('|valid|:', len(self.valid))

        if args.eval_percent < 1:
            self.test = np.random.randint(0, self.num_test,
                    size=(int(self.num_test * args.eval_percent,)))
            self.test += self.num_train + self.num_valid
        else:
            self.test = np.arange(self.num_train + self.num_valid, self.g.number_of_edges())
        print('|test|:', len(self.test))

    def get_edges(self, eval_type):
        """ Get all edges in this dataset

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing

        Returns
        -------
        np.array
            Edges
        """
        if eval_type == 'valid':
            return self.valid
        elif eval_type == 'test':
            return self.test
        else:
            raise Exception('get invalid type: ' + eval_type)

    def create_sampler(self, eval_type, batch_size, neg_sample_size, neg_chunk_size,
                       filter_false_neg, mode='head', num_workers=32, rank=0, ranks=1):
        """Create sampler for validation or testing

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        filter_false_neg : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        rank : int
            Which partition to sample.
        ranks : int
            Total number of partitions.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        edges = self.get_edges(eval_type)
        beg = edges.shape[0] * rank // ranks
        end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])
        edges = edges[beg: end]
        return EvalSampler(self.g, edges, batch_size, neg_sample_size, neg_chunk_size,
                           mode, num_workers, filter_false_neg)

class EvalDatasetNew(object):
    """Dataset for validation or testing

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    """
    def __init__(self, dataset, args, rel_constrain=False):
        src = [dataset.train[0]]
        etype_id = [dataset.train[1]]
        dst = [dataset.train[2]]
        self.n_entities = dataset.n_entities
        self.entity2id = dataset.entity2id
        self.relation2id = dataset.relation2id
        self.id2relation = {i:rel for rel,i in self.relation2id.items()}
        self.args = args

        print(type(dataset.train))
        print(type(dataset.valid))
        print(type(dataset.test))

        self.num_train = len(dataset.train[0])

        if not rel_constrain:
            if dataset.valid is not None:
                src.append(dataset.valid[0])
                etype_id.append(dataset.valid[1])
                dst.append(dataset.valid[2])
                self.num_valid = len(dataset.valid[0])
            else:
                self.num_valid = 0
        else:
            self.num_valid = 0

            
        if dataset.test is not None:
            src.append(dataset.test[0])
            etype_id.append(dataset.test[1])
            dst.append(dataset.test[2])
            self.num_test = len(dataset.test[0])
        else:
            self.num_test = 0
        #TODO: add this line later if needed
        # assert len(src) > 1, "we need to have at least validation set or test set."
        src = np.concatenate(src)
        etype_id = np.concatenate(etype_id)
        dst = np.concatenate(dst)

        print('src = {}'.format(src.shape))
        print('etype_id = {}'.format(etype_id.shape))
        print('dst = {}'.format(dst.shape))
        print('len(dataset.entity2id) = {}'.format(len(dataset.entity2id)))
        print('len(dataset.relation2id) = {}'.format(len(dataset.relation2id)))

        np.set_printoptions(threshold=sys.maxsize)
        # print('src = {}'.format(src))
        # print('dst = {}'.format(dst))
        # np.save('src.npy', src)
        # np.save('dst.npy', dst)
        
        print('n_entities = {}'.format(dataset.n_entities))

        print(np.amax(src))
        print(np.amin(src))

        print(np.amax(dst))
        print(np.amin(dst))

        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
                                    shape=[dataset.n_entities, dataset.n_entities])

        print('coo.shape = {}'.format(coo.shape))
        g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
        g.edata['tid'] = F.tensor(etype_id, F.int64)
        self.g = g

        self.valid = np.arange(self.num_train, self.num_train + self.num_valid)
        print('|valid|:', len(self.valid))

        self.test = []
        # self.test_old = np.arange(self.num_train + self.num_valid, self.g.number_of_edges())
        # print('|test|:', len(self.test))
        # TODO: assign self.test according to NEW dataset

        print('Reading test triples....')
        # heads = []
        # tails = []
        # rels = []
        triples = set()
        
        with open(args.test_triples_file) as f:
            for line in f:
                triple = line.strip().split('\t')
                h, r, t = triple[0], triple[1], triple[2]
                # heads.append(dataset.entity2id[h])
                # rels.append(dataset.relation2id[r])
                # tails.append(dataset.entity2id[t])

                triples.add((dataset.entity2id[h], dataset.relation2id[r], dataset.entity2id[t]))

        # heads = np.array(heads, dtype=np.int64)
        # tails = np.array(tails, dtype=np.int64)
        # rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} triples.'.format(len(triples)))
        
        for i, (h, r, t) in enumerate(tqdm(zip(src, etype_id, dst))):
            # print('h = {}, r = {}, t = {}'.format(h, r, t))
            if (h, r, t) in triples:
                self.test.append(i)
        
        self.test = np.asarray(self.test)
        # print((self.test - self.test_old).sum())
        if rel_constrain:
            self.test_rel_dict = {}

            for i, (h, r, t) in enumerate(tqdm(zip(src, etype_id, dst))):
                # print('h = {}, r = {}, t = {}'.format(h, r, t))
                if (h, r, t) in triples: # if (h,r,t) is in test triples
                    if r in self.test_rel_dict:
                        self.test_rel_dict[r].append(i)
                        # self.test_rel_dict[r].append((h, r, t))
                    else:
                        self.test_rel_dict[r] = [i]
            '''
            # create prob table
            self.rel_dict = {}

            for i, (h, r, t) in enumerate(tqdm(zip(src, etype_id, dst))):
                # print('h = {}, r = {}, t = {}'.format(h, r, t))
                if r in self.rel_dict:
                    self.rel_dict[r].append((h, r, t))
                else:
                    self.rel_dict[r] = [(h, r, t)]
            '''
            # print(self.test_rel_dict)
            # sys.exit(0)

    def get_edges(self, eval_type):
        """ Get all edges in this dataset

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing

        Returns
        -------
        np.array
            Edges
        """
        if eval_type == 'valid':
            return self.valid
        elif eval_type == 'test':
            return self.test
        else:
            raise Exception('get invalid type: ' + eval_type)

    def create_sampler(self, eval_type, batch_size, neg_sample_size, neg_chunk_size,
                       filter_false_neg, mode='head', num_workers=32, rank=0, ranks=1):
        """Create sampler for validation or testing

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        filter_false_neg : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        rank : int
            Which partition to sample.
        ranks : int
            Total number of partitions.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        edges = self.get_edges(eval_type)
        beg = edges.shape[0] * rank // ranks
        end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])
        edges = edges[beg: end]
        return EvalSampler(self.g, edges, batch_size, neg_sample_size, neg_chunk_size,
                           mode, num_workers, filter_false_neg)

    def create_rel_constrain_sampler(self, eval_type, batch_size, neg_sample_size, neg_chunk_size,
                       filter_false_neg, ent_child_dict, rel_type_dict,mode='head', num_workers=32, rank=0, ranks=1):
        assert self.test_rel_dict is not None

        eval_sampler_dict = {}
        sampler_prob_dict = {}

        if os.path.exists(os.path.join(self.args.data_path, 'sampler_prob_dict_{}.pickle'.format(mode))):
            sampler_prob_dict = pickle.load(open(os.path.join(self.args.data_path, 'sampler_prob_dict_{}.pickle'.format(mode)), 'rb'))

        for r in self.test_rel_dict:
            edge_triples = torch.tensor(self.test_rel_dict[r])

            '''
            src = np.asarray([x[0] for x in edge_triples])
            dst = np.asarray([x[2] for x in edge_triples])
            etype_id = np.asarray([x[1] for x in edge_triples])
            edges = torch.tensor(np.arange(len(edge_triples)))
            print('len(edge_triples) = {}'.format(len(edge_triples)))

            coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[self.n_entities, self.n_entities])
            g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
            g.edata['tid'] = F.tensor(etype_id, F.int64)
            '''
            print('mode = {}'.format(mode))
            
            # rel_triples_all = self.rel_dict[r]
            # src = np.asarray([x[0] for x in rel_triples_all])
            # dst = np.asarray([x[2] for x in rel_triples_all])
            if r not in sampler_prob_dict:
                if 'head' in mode:
                    # print([ent for ent_type in rel_type_dict[self.id2relation[r]]['head_types'] for ent in ent_child_dict[ent_type]])
                    src = np.asarray([self.entity2id[ent] for ent_type in rel_type_dict[self.id2relation[r]]['head_types'] for ent in ent_child_dict[ent_type] if ent in self.entity2id])
                    # print('src = {}'.format(src))
                    probs = torch.zeros((self.n_entities))
                    probs[src] = 1.0
                    probs = probs/torch.sum(probs).item()
                    print('probs = {}'.format(probs))
                else:
                    dst = np.asarray([self.entity2id[ent] for ent_type in rel_type_dict[self.id2relation[r]]['tail_types'] for ent in ent_child_dict[ent_type] if ent in self.entity2id])
                    # print('dst = {}'.format(dst))
                    probs = torch.zeros((self.n_entities))
                    probs[dst] = 1.0
                    probs = probs/torch.sum(probs).item()
                    print('probs = {}'.format(probs))

                sampler_prob_dict[r] = probs
            else:
                probs = sampler_prob_dict[r]
            
            print('flag 1')
            # neg_sample_size = 20
            neg_sample_size = min(1000, np.nonzero(probs).shape[0])
            print('neg_sample_size = {}'.format(neg_sample_size))
            eval_sampler_dict[r] = EvalSamplerWithConstrain(self.g, edge_triples, probs, batch_size, neg_sample_size, neg_chunk_size, mode, num_workers, filter_false_neg)
            print('flag 2')

        if not os.path.exists(os.path.join(self.args.data_path, 'sampler_prob_dict_{}.pickle'.format(mode))):
            pickle.dump(sampler_prob_dict, open(os.path.join(self.args.data_path, 'sampler_prob_dict_{}.pickle'.format(mode)), 'wb'))

        return eval_sampler_dict

class NewBidirectionalOneShotIterator:
    """Grouped samper iterator

    Parameters
    ----------
    dataloader_head : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in head mode
    dataloader_tail : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in tail mode
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    neg_sample_size : int
        How many negative edges sampled for each node.
    is_chunked : bool
        If True, the sampled batch is chunked.
    num_nodes : int
        Total number of nodes in the whole graph.
    """
    def __init__(self, dataloader_head, dataloader_tail, neg_chunk_size, neg_sample_size,
                 is_chunked, num_nodes):
        self.sampler_head = dataloader_head
        self.sampler_tail = dataloader_tail
        self.iterator_head = self.one_shot_iterator(dataloader_head, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    True, num_nodes)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    False, num_nodes)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            pos_g, neg_g = next(self.iterator_head)
        else:
            pos_g, neg_g = next(self.iterator_tail)
        return pos_g, neg_g

    @staticmethod
    def one_shot_iterator(dataloader, neg_chunk_size, neg_sample_size, is_chunked,
                          neg_head, num_nodes):
        while True:
            for pos_g, neg_g in dataloader:
                neg_g = create_neg_subgraph(pos_g, neg_g, neg_chunk_size, neg_sample_size,
                                            is_chunked, neg_head, num_nodes)
                if neg_g is None:
                    continue

                pos_g.ndata['id'] = pos_g.parent_nid
                neg_g.ndata['id'] = neg_g.parent_nid
                pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
                yield pos_g, neg_g
