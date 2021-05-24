import sys, os
import argparse
import numpy as np
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
from wiki2vec.dictionary import Dictionary
import gensim
from gensim import utils

'''
convert the word embeddings only from skip-gram checkpoint file to word2vec format
'''

def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).

    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with utils.open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sg-ckpt-file', type=str, required=True)
    parser.add_argument('--dict-file', type=str, required=True)
    parser.add_argument('--out-file', type=str, required=True)
    parser.add_argument('--entity', action='store_true')
    parser.add_argument('--both', action='store_true')

    args = parser.parse_args()

    dictionary = Dictionary.load(args.dict_file)
    sg_model_emb0 = np.load(args.sg_ckpt_file) # [vocab_size, embed_dim]

    if not args.both:
        if not args.entity:
            sg_model_emb0 = sg_model_emb0[:dictionary.word_size, :]
            assert sg_model_emb0.shape[0] == dictionary.word_size
        else:
            sg_model_emb0 = sg_model_emb0[dictionary.word_size:, :]
            assert sg_model_emb0.shape[0] == dictionary.entity_size
    else:
        assert sg_model_emb0.shape[0] == dictionary.word_size + dictionary.entity_size

    print(sg_model_emb0.shape)
    embed_dim = sg_model_emb0.shape[1]

    # print([x.text for x in dictionary.words()])
    # print([x.title for x in dictionary.entities()])
    print('flag 0')

    if not args.both:
        if not args.entity:
            d = {dictionary.get_word_by_index(i).text : sg_model_emb0[i,:] for i in range(dictionary.word_size)}
        else:
            d = {dictionary.get_entity_by_index(i + dictionary.word_size).title.replace(' ','_') : sg_model_emb0[i,:] for i in range(dictionary.entity_size)}
    else:
        d = {}
        for i in range(dictionary.word_size+dictionary.entity_size):
            if i < dictionary.word_size:
                d[dictionary.get_word_by_index(i).text] = sg_model_emb0[i,:]
            else:
                d[dictionary.get_entity_by_index(i).title.replace(' ','_')] = sg_model_emb0[i,:]

    # print(d.keys())
    print('flag 1')

    m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=embed_dim)
    m.vocab = d
    m.vectors = np.array(list(d.values()))
    print('flag 2')
    # print(m.vocab.keys())

    my_save_word2vec_format(binary=True, fname=args.out_file, total_vec=len(d), vocab=m.vocab, vectors=m.vectors)


