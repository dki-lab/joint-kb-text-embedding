# kb-text-embed-study

## Installation

### Install Python dependencies
```
pip install -r requirements.txt
```

### Install wiki2vec extension
TODO

## Data preprocessing

### Download data
```
# download wikidata Dec 2020 processed json files
TODO

# download wikidata triple files
TODO

# download wikipedia raw dump file in WIKIPEDIA_PROC_DATA
TODO

# download Few-shot link prediction dataset
TODO

# download Analogical Reasoning dataset
TODO
```

### Pre-process data
This step is not needed if you download the pre-processed data as above.

#### Pre-process wikidata
```
mkdir $WIKIDATA_PROC_JSON_DIR
mkdir $WIKIDATA_TRIPLES_DIR
python utils/create_proc_wikidata.py --input_json_file $RAW_WIKIDATA_JSON_FILE --out_dir $WIKIDATA_PROC_JSON_DIR
python utils/generate_triples.py $WIKIDATA_PROC_JSON_DIR $WIKIDATA_TRIPLES_DIR/triples.tsv

# We shuffle triples.tsv and split it into train-valid-test files (wikidata_train.tsv wikidata_valid.tsv wikidata_test.tsv) in the ratio 0.85:0.075:0.075.
```

#### Pre-process wikipedia raw dump
```
wikipedia2vec build-dump-db $WIKIPEDIA_PROC_DATA/DUMP_FILE $WIKIPEDIA_PROC_DATA/db_file
wikipedia2vec build-dictionary $WIKIPEDIA_PROC_DATA/db_file $WIKIPEDIA_PROC_DATA/dict_file
wikipedia2vec build-link-graph $WIKIPEDIA_PROC_DATA/db_file $WIKIPEDIA_PROC_DATA/dict_file $WIKIPEDIA_PROC_DATA/link_graph_file
wikipedia2vec build-mention-db $WIKIPEDIA_PROC_DATA/db_file $WIKIPEDIA_PROC_DATA/dict_file $WIKIPEDIA_PROC_DATA/mentiondb_file
```

#### Create Few-shot link prediction dataset
TODO

#### Create Analogical Reasoning dataset
TODO

## Experiments

### Setup environment variables
Set the environment variables `WIKIDATA_TRIPLES_DIR`, `WIKIDATA_PROC_JSON_DIR`, `WIKIPEDIA_PROC_DATA`, `SAVE_DIR`, `$TEST_TRIPLES_FILE`, `WIKIDATA_MARCH_2020_TRIPLES_DIR` (if applicable).

### Run training for a particular KB-text alignment method 
```
python -u train.py --model_name TransE_l2 --batch_size 1000 --log_interval 10000 --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 300 --gamma 19.9 --lr 0.25 --batch_size_eval 16 --data_path $WIKIDATA_TRIPLES_DIR --data_files wikidata_train.tsv wikidata_valid.tsv wikidata_test.tsv --format raw_udd_hrt --dump-db-file $WIKIPEDIA_PROC_DATA/db_file --dictionary-file $WIKIPEDIA_PROC_DATA/dict_file --mention-db-file $WIKIPEDIA_PROC_DATA/mentiondb_file --link-graph-file $WIKIPEDIA_PROC_DATA/link_graph_file --num_thread 1 --valid --test --neg_deg_sample --neg_deg_sample_eval --neg_sample_size_eval 1000 --no_eval_filter --save_path $SAVE_DIR --reg_coeff 0.1 --reg-loss-start-epoch 0 --n_iters 1 --num_proc 4 --num_proc_train 4 --timeout 200 --wiki-link-file $WIKIDATA_PROC_JSON_DIR/wikipedia_links.json
```

### Run evaluation for KB embeddings
```
python -u eval.py --model_name TransE_l2 --hidden_dim 300 --gamma 19.9 --batch_size_eval 16 --data_path $DATA_DIR --data_files wikidata_train.tsv wikidata_valid.tsv wikidata_test.tsv --format raw_udd_hrt --num_thread 1 --num_proc 8 --neg_sample_size_eval 1000 --test-triples-file $TEST_TRIPLES_FILE --model_path $SAVE_DIR/
```

### Run link prediction evaluation for KB-text aligned embeddings
```
python -u eval.py --model_name TransE_l2 --hidden_dim 300 --gamma 19.9 --batch_size_eval 16 --data_path $DATA_DIR --data_files wikidata_train.tsv wikidata_valid.tsv wikidata_test.tsv --format raw_udd_hrt --num_thread 1 --num_proc 8 --neg_sample_size_eval 1000 --test-triples-file $TEST_TRIPLES_FILE --model_path $SAVE_DIR/ --dictionary-file $WIKIPEDIA_PROC_DATA/dict_file --wiki-link-file ~/data/wikidata_dec_20_proc_json/wikipedia_links.json
```

### Run link prediction evaluation for KB-text aligned embeddings with type constraints
```
python -u eval_type_constraint.py --model_name TransE_l2 --hidden_dim 300 --gamma 19.9 --batch_size_eval 16 --data_path $DATA_DIR --data_files wikidata_train.tsv wikidata_valid.tsv wikidata_test.tsv --format raw_udd_hrt --num_thread 1 --num_proc 8 --neg_sample_size_eval 1000 --test-triples-file $TEST_TRIPLES_FILE --model_path $SAVE_DIR/ --rel-type-dict-file $WIKIDATA_MARCH_2020_TRIPLES_DIR/rel_type_dict.pickle --entity-child-dict-file $WIKIDATA_MARCH_2020_TRIPLES_DIR/entity_child_dict.json --sampler-type both
```
