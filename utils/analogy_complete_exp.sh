#!/bin/sh

PYTHONHASHSEED=1234 python utils/convert_sg_to_word2vec_format.py --sg-ckpt-file $SAVE_DIR/FB15k_TransE_l2_emb0_sg.npy --dict-file $WIKIPEDIA_PROC_DATA/dict_file --out-file $SAVE_DIR/FB15k_TransE_l2_emb0_sg_ent.bin --entity

mkdir $ANALOGY_DATASET_DIR/results

for rel in P103 P1040 P105 P1196 P119 P123 P137 P138 P140 P1412
do
	PYTHONHASHSEED=1234 python utils/eval_analogy.py --sg-ckpt-file $SAVE_DIR/FB15k_TransE_l2_emb0_sg_ent.bin --analogy-file $ANALOGY_DATASET_DIR/${rel}.txt --entity-counter-file $WIKIPEDIA_PROC_DATA/ent_counter_names.json --wiki-link-file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json --entity-file $WIKIDATA_TRIPLES_DIR/entities.tsv --degree-sampling > $ANALOGY_DATASET_DIR/results/${rel}.txt &
done

sleep 4m

for rel in P1532 P155 P156 P159 P170 P171 P172 P179 P17 P19
do
	PYTHONHASHSEED=1234 python utils/eval_analogy.py --sg-ckpt-file $SAVE_DIR/FB15k_TransE_l2_emb0_sg_ent.bin --analogy-file $ANALOGY_DATASET_DIR/${rel}.txt --entity-counter-file $WIKIPEDIA_PROC_DATA/ent_counter_names.json --wiki-link-file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json --entity-file $WIKIDATA_TRIPLES_DIR/entities.tsv --degree-sampling > $ANALOGY_DATASET_DIR/results/${rel}.txt &
done

sleep 4m

for rel in P206 P20 P22 P241 P25 P26 P276 P279 P27 P282
do
	PYTHONHASHSEED=1234 python utils/eval_analogy.py --sg-ckpt-file $SAVE_DIR/FB15k_TransE_l2_emb0_sg_ent.bin --analogy-file $ANALOGY_DATASET_DIR/${rel}.txt --entity-counter-file $WIKIPEDIA_PROC_DATA/ent_counter_names.json --wiki-link-file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json --entity-file $WIKIDATA_TRIPLES_DIR/entities.tsv --degree-sampling > $ANALOGY_DATASET_DIR/results/${rel}.txt &
done

sleep 4m

for rel in P344 P364 P36 P407 P410 P413 P462 P495 P509 P53
do
	PYTHONHASHSEED=1234 python utils/eval_analogy.py --sg-ckpt-file $SAVE_DIR/FB15k_TransE_l2_emb0_sg_ent.bin --analogy-file $ANALOGY_DATASET_DIR/${rel}.txt --entity-counter-file $WIKIPEDIA_PROC_DATA/ent_counter_names.json --wiki-link-file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json --entity-file $WIKIDATA_TRIPLES_DIR/entities.tsv --degree-sampling > $ANALOGY_DATASET_DIR/results/${rel}.txt &
done

sleep 4m

for rel in P57 P641 P6886 P734 P735 P750 P7938 P840 P86 P97
do
	PYTHONHASHSEED=1234 python utils/eval_analogy.py --sg-ckpt-file $SAVE_DIR/FB15k_TransE_l2_emb0_sg_ent.bin --analogy-file $ANALOGY_DATASET_DIR/${rel}.txt --entity-counter-file $WIKIPEDIA_PROC_DATA/ent_counter_names.json --wiki-link-file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json --entity-file $WIKIDATA_TRIPLES_DIR/entities.tsv --degree-sampling > $ANALOGY_DATASET_DIR/results/${rel}.txt &
done

