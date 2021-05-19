import sys, os
import json
import codecs
from tqdm import tqdm
from collections import Counter

if __name__=='__main__':
	wikidata_json_dir = sys.argv[1]
	out_file = sys.argv[2]

	f1 = open(out_file, 'w')
	entity_counter = Counter()
	relation_counter = Counter()
	
	entity_threshold = 10
	relation_threshold = 5

	# update entity and relation counters
	for json_file in os.listdir(wikidata_json_dir):
		if 'comp_wikidata' not in json_file:
			continue
		wikidata_dict = json.load(codecs.open(os.path.join(wikidata_json_dir, json_file), 'r', 'utf-8'))

		for head_entity in tqdm(wikidata_dict.keys()):
			if not head_entity.startswith('Q'):
				continue
			for relation_id in wikidata_dict[head_entity]['predicates']:
				if not relation_id.startswith('P'):
					continue
				for tail_entity in wikidata_dict[head_entity]['predicates'][relation_id]:
					if not tail_entity.startswith('Q'):
						continue
					entity_counter.update([head_entity])
					entity_counter.update([tail_entity])
					relation_counter.update([relation_id])

		del wikidata_dict

	n_valid_entities = len(list(filter(lambda x:x[1]>=entity_threshold, entity_counter.most_common())))
	n_valid_relations = len(list(filter(lambda x:x[1]>=relation_threshold, relation_counter.most_common())))

	print('n_valid_entities = {}'.format(n_valid_entities))
	print('n_valid_relations = {}'.format(n_valid_relations))


	# write triples to file
	for json_file in os.listdir(wikidata_json_dir):
		if 'comp_wikidata' not in json_file:
			continue
		wikidata_dict = json.load(codecs.open(os.path.join(wikidata_json_dir, json_file), 'r', 'utf-8'))

		for head_entity in tqdm(wikidata_dict.keys()):
			if not head_entity.startswith('Q'):
				continue
			for relation_id in wikidata_dict[head_entity]['predicates']:
				if not relation_id.startswith('P'):
					continue
				for tail_entity in wikidata_dict[head_entity]['predicates'][relation_id]:
					if not tail_entity.startswith('Q'):
						continue
					if entity_counter[head_entity] >= entity_threshold and 	entity_counter[tail_entity] >= entity_threshold	and relation_counter[relation_id] >= relation_threshold:
						f1.write('{}\t{}\t{}\n'.format(head_entity, relation_id, tail_entity))

		f1.flush()

		del wikidata_dict

	f1.close()

