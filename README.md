# joint-kb-text-embedding

Code for the ACL 2021 paper **A Systematic Investigation of KB-Text Embedding Alignment at Scale**  
**Authors**: Vardaan Pahuja, Yu Gu, Wenhu Chen, Mehdi Bahrami, Lei Liu, Wei-Peng Chen and Yu Su  
This implementation is based on the [DGL-KE](https://github.com/awslabs/dgl-ke) and [Wikipedia2Vec](https://github.com/wikipedia2vec/wikipedia2vec) libraries.

## Installation

### Install Python dependencies
```
pip install -r requirements.txt
```

### Install wiki2vec extension
```
cd wikinew/
./cythonize.sh
python setup.py install
```

## Data preprocessing

### Download data

1. Download Wikidata Dec. 2020 triple files from [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/EtvIP8Hyo6pIlgIEQbCsrwMBJAk9pf7SMooynsUkdzWBoA) and store directory path into environment variable `WIKIDATA_TRIPLES_DIR`

2. Download pre-processed Wikipedia files from [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/EoT_yv2sKbFPj_RzhiyZc2wB9VNXL5lz6ExZ7tb7rwaW9A) and store directory path into environment variable `WIKIPEDIA_PROC_DATA`

3. Download Few-shot link prediction dataset from [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/EpfqthRPp9FLrERxCnXwPSEBNryTYDzyx_4_HQ1yFlc9cg) and store directory path into environment variable `WIKIDATA_FS_LP_DIR`

4. Download Analogical Reasoning dataset from [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/EgfwJgbJFGhKiuq7chlV7AkBl8oqW4N2qvzScMzHEfIlHA?e=E2h1GR) and store directory path into environment variable `ANALOGY_DATASET_DIR`

5. Download `wikipedia_links.json` from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/Efz-xTWWiXtGmT8n7MPzuh0BKSDbT6a5hWoo0lfm3elb7g) and save it in the dir corresponding to the environment variable `WIKIDATA_PROC_JSON_DIR`.

6. Download `rel_type_dict.pickle` from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/EREYiat2yeRDq4l4YGuCVhUBwoBVhLtDzS29A6hMRn89Pg?e=IgoijP) and save it in the dir corresponding to the environment variable `WIKIDATA_PROC_JSON_DIR`.

7. Download `entity_child_dict.pickle` from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/EcEpxy7hnQBMjQdrFFUscJcBW63iSmxKN3OHerWJIiDR3g) and save it in the dir corresponding to the environment variable `WIKIDATA_PROC_JSON_DIR`.

8. Download `ent_counter_names.json` from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/ESlMMt3K-5tImUO20Y9mQ84BbLIPfaNzBNdLmYePV4K0dQ) and save it in the dir corresponding to the environment variable `WIKIDATA_PROC_JSON_DIR`.

9. Download Wikidata March 2020 triple files from [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/EqpDSZAiVwlPm8w1UsXrbfABUEQzleegPiJhTsxjlxAa9A) and store directory path into environment variable `WIKIDATA_MAR_20_TRIPLES_DIR`

10. Download the COVID case-study triples dir from [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/ErvVsCsfdIxMgybmKfFv_zkBKcl5yEa6_Lg1iRVIJk1mJQ) and and store the dir name in the environment variable `COVID_TRIPLES_DIR`.

### Pre-process data
This step is not needed if you download the pre-processed data as above.

#### Download raw Wikidata and Wikipedia dumps

1. Download Wikidata raw dump file from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/ESyvetnF6KpKvSAaQTWJoq4B8bp2sCBmK6awzMapcOyYcg) and set environment variable `RAW_WIKIDATA_JSON_FILE` to its path.
2. Download Wikipedia raw dump file from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/ERXwyla6Qn9IioXUFpV1x3EBpHvXEwIb22IZlOP29xDnxQ) and set environment variable `DUMP_FILE` to its path.


#### Pre-process Wikidata
```
mkdir $WIKIDATA_PROC_JSON_DIR
mkdir $WIKIDATA_TRIPLES_DIR
python utils/create_proc_wikidata.py --input_json_file $RAW_WIKIDATA_JSON_FILE --out_dir $WIKIDATA_PROC_JSON_DIR
python utils/generate_triples.py $WIKIDATA_PROC_JSON_DIR $WIKIDATA_TRIPLES_DIR/triples.tsv

# We shuffle triples.tsv and split it into train-valid-test files (wikidata_train.tsv wikidata_valid.tsv wikidata_test.tsv) in the ratio 0.85:0.075:0.075.

python utils/create_wikipedia_wikidata_link_dict.py --input_json_file $RAW_WIKIDATA_JSON_FILE --out_links_file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json
python utils/create_entity_type_dict.py --wikidata-triples-file $WIKIDATA_TRIPLES_DIR/wikidata_train.tsv --out-dir $WIKIDATA_PROC_JSON_DIR
python utils/create_rel_type_dict.py --wikidata-triples-file $WIKIDATA_TRIPLES_DIR/wikidata_train.tsv --entity-type-dict-file $WIKIDATA_PROC_JSON_DIR/entity_type_dict.json --out-dir $WIKIDATA_PROC_JSON_DIR
python utils/create_counter_domain_intersection.py --triples-file $WIKIDATA_TRIPLES_DIR/wikidata_train.tsv --wiki-link-file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json --entity-file $WIKIDATA_TRIPLES_DIR/entities.tsv --dict-file $WIKIPEDIA_PROC_DATA/dict_file --out-dir $WIKIDATA_PROC_JSON_DIR
```

#### Pre-process wikipedia raw dump
```
mkdir $WIKIPEDIA_PROC_DATA
wikipedia2vec build-dump-db $DUMP_FILE $WIKIPEDIA_PROC_DATA/db_file
wikipedia2vec build-dictionary $WIKIPEDIA_PROC_DATA/db_file $WIKIPEDIA_PROC_DATA/dict_file
wikipedia2vec build-link-graph $WIKIPEDIA_PROC_DATA/db_file $WIKIPEDIA_PROC_DATA/dict_file $WIKIPEDIA_PROC_DATA/link_graph_file
wikipedia2vec build-mention-db $WIKIPEDIA_PROC_DATA/db_file $WIKIPEDIA_PROC_DATA/dict_file $WIKIPEDIA_PROC_DATA/mentiondb_file
```

<!-- #### Create Few-shot link prediction dataset
TODO

#### Create Analogical Reasoning dataset
TODO -->

## Experiments

### Few-shot Link Prediction

#### Setup environment variables
Set the environment variables `WIKIDATA_FS_LP_DIR`, `WIKIDATA_PROC_JSON_DIR`, `WIKIPEDIA_PROC_DATA`, `SAVE_DIR`, $BALANCE_PARAM and navigate to the directory of the desired kb-text alignment method.

#### Run training for train set (Full)
```
python train.py --model_name TransE_l2 --batch_size 1000 --log_interval 10000 --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 300 --gamma 19.9 --lr 0.25 --batch_size_eval 16 --data_path $WIKIDATA_FS_LP_DIR --data_files wikidata_train_full.tsv wikidata_test.tsv wikidata_test.tsv --format raw_udd_hrt --dump-db-file $WIKIPEDIA_PROC_DATA/db_file --dictionary-file $WIKIPEDIA_PROC_DATA/dict_file --mention-db-file $WIKIPEDIA_PROC_DATA/mention_db_file --link-graph-file $WIKIPEDIA_PROC_DATA/link_graph_file --num_thread 1 --neg_deg_sample --save_path $SAVE_DIR --balance_param $BALANCE_PARAM --reg-loss-start-epoch 0 --n_iters 20 --num_proc 8 --num_proc_train 32 --timeout 200 --wiki-link-file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json
```

#### Run training for train set (support)

```
python train.py --model_name TransE_l2 --batch_size 1000 --log_interval 10000 --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 300 --gamma 19.9 --lr 0.25 --batch_size_eval 16 --data_path $WIKIDATA_FS_LP_DIR --data_files wikidata_train_support.tsv wikidata_test.tsv wikidata_test.tsv --format raw_udd_hrt --dump-db-file $WIKIPEDIA_PROC_DATA/db_file --dictionary-file $WIKIPEDIA_PROC_DATA/dict_file --mention-db-file $WIKIPEDIA_PROC_DATA/mention_db_file --link-graph-file $WIKIPEDIA_PROC_DATA/link_graph_file --num_thread 1 --neg_deg_sample --save_path $SAVE_DIR --balance_param $BALANCE_PARAM --reg-loss-start-epoch 0 --n_iters 20 --num_proc 8 --num_proc_train 32 --timeout 200 --wiki-link-file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json
```

#### Run link prediction evaluation for Test set (Both in support)
```
python eval_type_constraint.py --model_name TransE_l2 --hidden_dim 300 --gamma 19.9 --batch_size_eval 16 --data_path $WIKIDATA_FS_LP_DIR --data_files wikidata_train_full.tsv wikidata_test.tsv wikidata_test.tsv --format raw_udd_hrt --num_thread 1 --num_proc 1 --neg_sample_size_eval 1000 --test-triples-file $WIKIDATA_FS_LP_DIR/wikidata_test_support.tsv --model_path $SAVE_DIR/ --rel-type-dict-file $WIKIDATA_PROC_JSON_DIR/rel_type_dict.pickle --entity-child-dict-file $WIKIDATA_PROC_JSON_DIR/entity_child_dict.json --sampler-type both
```

#### Run link prediction evaluation for Test set (Missing support)
```
python eval_type_constraint.py --model_name TransE_l2 --hidden_dim 300 --gamma 19.9 --batch_size_eval 16 --data_path $WIKIDATA_FS_LP_DIR --data_files wikidata_train_full.tsv wikidata_test.tsv wikidata_test.tsv --format raw_udd_hrt --num_thread 1 --num_proc 1 --neg_sample_size_eval 1000 --test-triples-file $WIKIDATA_FS_LP_DIR/wikidata_test_missing_support.tsv --model_path $SAVE_DIR/ --rel-type-dict-file $WIKIDATA_PROC_JSON_DIR/rel_type_dict.pickle --entity-child-dict-file $WIKIDATA_PROC_JSON_DIR/entity_child_dict.json --sampler-type both
```

### Analogical Reasoning

#### Setup environment variables
Set the environment variables `WIKIDATA_TRIPLES_DIR`, `WIKIDATA_PROC_JSON_DIR`, `WIKIPEDIA_PROC_DATA`, `SAVE_DIR`, $BALANCE_PARAM and navigate to the directory of the desired kb-text alignment method.

#### Run training

```
python train.py --model_name TransE_l2 --batch_size 1000 --log_interval 10000 --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 300 --gamma 19.9 --lr 0.25 --batch_size_eval 16 --data_path $WIKIDATA_TRIPLES_DIR --data_files wikidata_train.tsv wikidata_valid.tsv wikidata_test.tsv --format raw_udd_hrt --dump-db-file $WIKIPEDIA_PROC_DATA/db_file --dictionary-file $WIKIPEDIA_PROC_DATA/dict_file --mention-db-file $WIKIPEDIA_PROC_DATA/mention_db_file --link-graph-file $WIKIPEDIA_PROC_DATA/link_graph_file --num_thread 1 --neg_deg_sample --save_path $SAVE_DIR --balance_param $BALANCE_PARAM --reg-loss-start-epoch 0 --n_iters 20 --num_proc 8 --num_proc_train 32 --timeout 200 --wiki-link-file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json
```

#### Run evaluation
```
sh utils/analogy_complete_exp.sh
```

## Pre-trained embeddings

The pre-trained embeddings for each of the 4 alignment methods can be downloaded below. The description of the filenames is as follows:
1. TransE_l2_emb0_sg.npy: Skip-gram Embeddings for (words + entities), [Word ID to name mapping file](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/ER12jePBTp5DpnZqGQD2Oh8BRKLT4yqolAMoJQ8uoEcW9A) [Entity ID to name mapping file](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/EUZsnCwLyh5Nk_r6l9It1QsBnYOuM11mTeBtb3102ruaNQ)
2. TransE_l2_entity.npy: TransE embeddings for entities, [Entity ID to name mapping file](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/EUZsnCwLyh5Nk_r6l9It1QsBnYOuM11mTeBtb3102ruaNQ)
3. TransE_l2_relation.npy: TransE embeddings for relations, [Relation ID to name mapping file](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/ERU4oc2V6VVOhqZGs6t9aucBp3Nx5dd78PaGT7Nm13gMrg)

### Alignment using Entity Names (balance param.=1.0)
[Download Link](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/Es-hgly0a8hOsWBkZ4mOa9ABlhW02WZVaXea9ho53_YgUg)

### Same Embedding alignment
[Download Link](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/ElxfuCrsVWJNrU8Rrl6mMygBFU0JiMk67D1MNrCeL8uZTQ)

### Projection alignment (balance param.=1e-3)
[Download Link](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/Ej_xzDLRCBhFoSA8mCI4UbUBpwWHg8ZfHQIFR2BlUFuyHA)

### Alignment using Wikipedia Anchors (balance param.=1.0)
[Download Link](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/EvxuXCCSvDNFtfWXa9SIqV4BLQqOiBv7EQzdOQnfGj34Hw)

## Covid case-study

#### Run training

```
python train.py --model_name TransE_l2 --batch_size 1000 --log_interval 10000 --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 300 --gamma 19.9 --lr 0.25 --batch_size_eval 16 --data_path $WIKIDATA_MAR_20_TRIPLES_DIR --data_files triples.tsv --format raw_udd_hrt --dump-db-file $WIKIPEDIA_PROC_DATA/db_file --dictionary-file $WIKIPEDIA_PROC_DATA/dict_file --mention-db-file $WIKIPEDIA_PROC_DATA/mention_db_file --link-graph-file $WIKIPEDIA_PROC_DATA/link_graph_file --num_thread 1 --neg_deg_sample --save_path $SAVE_DIR --balance_param $BALANCE_PARAM --reg-loss-start-epoch 0 --n_iters 20 --num_proc 8 --num_proc_train 32 --timeout 200 --wiki-link-file $WIKIDATA_MAR_20_PROC_JSON_DIR/wikipedia_links.json
```

#### Run Evaluation

```
# P5642 (Risk Factor)
python -u eval_type_constraint.py --model_name TransE_l2 --hidden_dim 300 --gamma 19.9 --batch_size_eval 1 --data_path $WIKIDATA_MAR_20_TRIPLES_DIR --data_files wikidata_train.tsv wikidata_test.tsv wikidata_test.tsv --format udd_hrt --num_thread 1 --num_proc 1 --neg_sample_size_eval 1000 --test-triples-file $COVID_TRIPLES_DIR/wikidata_test_covid_P5642.tsv --model_path $SAVE_DIR --rel-type-dict-file $WIKIDATA_MAR_20_PROC_JSON_DIR/rel_type_dict.pickle --entity-child-dict-file $WIKIDATA_MAR_20_PROC_JSON_DIR/entity_child_dict.json --sampler-type tail

# P780 (Symptoms)
python -u eval_type_constraint.py --model_name TransE_l2 --hidden_dim 300 --gamma 19.9 --batch_size_eval 1 --data_path $WIKIDATA_MAR_20_TRIPLES_DIR --data_files wikidata_train.tsv wikidata_test.tsv wikidata_test.tsv --format udd_hrt --num_thread 1 --num_proc 1 --neg_sample_size_eval 1000 --test-triples-file $COVID_TRIPLES_DIR/wikidata_test_covid_P780.tsv --model_path $SAVE_DIR --rel-type-dict-file $WIKIDATA_MAR_20_PROC_JSON_DIR/rel_type_dict.pickle --entity-child-dict-file $WIKIDATA_MAR_20_PROC_JSON_DIR/entity_child_dict.json --sampler-type tail

# P509 (Cause of death)
python -u eval_type_constraint.py --model_name TransE_l2 --hidden_dim 300 --gamma 19.9 --batch_size_eval 1 --data_path $WIKIDATA_MAR_20_TRIPLES_DIR --data_files wikidata_train.tsv wikidata_test.tsv wikidata_test.tsv --format udd_hrt --num_thread 1 --num_proc 1 --neg_sample_size_eval 1000 --test-triples-file $COVID_TRIPLES_DIR/wikidata_test_covid_P509.tsv --model_path $SAVE_DIR --rel-type-dict-file $WIKIDATA_MAR_20_PROC_JSON_DIR/rel_type_dict.pickle --entity-child-dict-file $WIKIDATA_MAR_20_PROC_JSON_DIR/entity_child_dict.json --sampler-type head

# P1050 (medical condition)
python -u eval_type_constraint.py --model_name TransE_l2 --hidden_dim 300 --gamma 19.9 --batch_size_eval 1 --data_path $WIKIDATA_MAR_20_TRIPLES_DIR --data_files wikidata_train.tsv wikidata_test.tsv wikidata_test.tsv --format udd_hrt --num_thread 1 --num_proc 1 --neg_sample_size_eval 1000 --test-triples-file $COVID_TRIPLES_DIR/wikidata_test_covid_P1050.tsv --model_path $SAVE_DIR --rel-type-dict-file $WIKIDATA_MAR_20_PROC_JSON_DIR/rel_type_dict.pickle --entity-child-dict-file $WIKIDATA_MAR_20_PROC_JSON_DIR/entity_child_dict.json --sampler-type head
```