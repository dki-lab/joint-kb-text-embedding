# kb-text-embed-study

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

1. Download Wikidata triple files from [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/EtvIP8Hyo6pIlgIEQbCsrwMBJAk9pf7SMooynsUkdzWBoA) and store directory path into environment variable `WIKIDATA_TRIPLES_DIR`

2. Download pre-processed Wikipedia files from [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/EoT_yv2sKbFPj_RzhiyZc2wB9VNXL5lz6ExZ7tb7rwaW9A) and store directory path into environment variable `WIKIPEDIA_PROC_DATA`

4. Download Few-shot link prediction dataset from [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/EpfqthRPp9FLrERxCnXwPSEBNryTYDzyx_4_HQ1yFlc9cg) and store directory path into environment variable `WIKIDATA_FS_LP_DIR`

5. Download Analogical Reasoning dataset from [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/EgfwJgbJFGhKiuq7chlV7AkBl8oqW4N2qvzScMzHEfIlHA?e=E2h1GR) and store directory path into environment variable `ANALOGY_DATASET_DIR`

6. Download `wikipedia_links.json` from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/Efz-xTWWiXtGmT8n7MPzuh0BKSDbT6a5hWoo0lfm3elb7g) and save it in the dir corresponding to the environment variable `WIKIDATA_PROC_JSON_DIR`.

7. Download `rel_type_dict.pickle` from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/EREYiat2yeRDq4l4YGuCVhUBwoBVhLtDzS29A6hMRn89Pg?e=IgoijP) and save it in the dir corresponding to the environment variable `WIKIDATA_PROC_JSON_DIR`.

8. Download `entity_child_dict.pickle` from [here]() and save it in the dir corresponding to the environment variable `WIKIDATA_PROC_JSON_DIR`.

### Pre-process data
This step is not needed if you download the pre-processed data as above.

#### Download raw Wikidata and Wikipedia dumps

1. Download Wikidata raw dump file from [here](https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz) and set environment variable `RAW_WIKIDATA_JSON_FILE` to its path.
2. Download Wikipedia raw dump file from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/pahuja_9_buckeyemail_osu_edu/ERXwyla6Qn9IioXUFpV1x3EBpHvXEwIb22IZlOP29xDnxQ) and set environment variable `DUMP_FILE` to its path.


#### Pre-process Wikidata
```
mkdir $WIKIDATA_PROC_JSON_DIR
mkdir $WIKIDATA_TRIPLES_DIR
python utils/create_proc_wikidata.py --input_json_file $RAW_WIKIDATA_JSON_FILE --out_dir $WIKIDATA_PROC_JSON_DIR
python utils/generate_triples.py $WIKIDATA_PROC_JSON_DIR $WIKIDATA_TRIPLES_DIR/triples.tsv

# We shuffle triples.tsv and split it into train-valid-test files (wikidata_train.tsv wikidata_valid.tsv wikidata_test.tsv) in the ratio 0.85:0.075:0.075.
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