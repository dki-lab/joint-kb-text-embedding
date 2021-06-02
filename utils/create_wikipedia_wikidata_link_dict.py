import codecs, json, sys, os
import logging
import argparse
import gzip
from tqdm import tqdm
import time

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_json_file', type=str, help='compressed wikidata json file')
	parser.add_argument('--out_links_file', type=str, help='file containing wikipedia links')

	args = parser.parse_args()

	inFp = gzip.open(args.input_json_file, 'r', 'utf-8')

	inFp.readline()
	count = 0
	not_present_count = 0

	wikipedia_link_dict = {}

	start_time = time.time()
	
	for line in inFp:
		try :
			line = line.decode("utf-8")
			if not line.strip().startswith('{') :
				continue
			count += 1
			if line.strip().endswith(',') :
				line = line[0:-2].strip()
			json_entry = line
			js = json.loads(json_entry)
			ID = js['id']

			if not ID.startswith('Q'):
				continue

			if 'claims' not in js :
				continue

			if 'sitelinks' in js and 'enwiki' in js['sitelinks'] and 'title' in js['sitelinks']['enwiki']:
				wiki_title = js['sitelinks']['enwiki']['title']
				wikipedia_link_dict[ID] = wiki_title

		except Exception as e:
			print('{}'.format(count))
			logging.exception("Something awful happened!")

	inFp.close()
	
	with codecs.open(args.out_links_file, 'w', 'utf-8') as g:
		json.dump(wikipedia_link_dict, g, indent=4)

