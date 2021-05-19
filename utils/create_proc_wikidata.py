import codecs, json, sys, os
import logging
import argparse
import gzip
from tqdm import tqdm

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_json_file', type=str, help='compressed wikidata json file')
	parser.add_argument('--out_dir', type=str, help='output dir. of processed json files')
	args = parser.parse_args()

	inFp = gzip.open(args.input_json_file, 'r', 'utf-8')

	inFp.readline()
	count=0

	outer_dict1={}
	outer_dict2={}
	outer_dict3={}
	outer_dict4={}
	outer_dict5={}
	outer_dict6={}

	item_dict = {}
	property_dict={}

	for line in tqdm(inFp):
		try :
			line = line.decode("utf-8")
			# print(type(line))
			if not line.strip().startswith('{') :
				continue
			count += 1
			if line.strip().endswith(',') :
				line = line[0:-2].strip()
			json_entry = line
			js = json.loads(json_entry)
			ID = js['id'] 

			q_dict={}
			prop_dict={}
			alias_list=[]
			
			if 'claims' not in js :
				continue
			for prop in js['claims'].keys():
				qlist=[]
				
				for q in range(len(js['claims'][prop])):
					if 'mainsnak' in js['claims'][prop][q] and 'datavalue' in js['claims'][prop][q]['mainsnak'] and 'value' in js['claims'][prop][q]['mainsnak']['datavalue'] and 'id' in js['claims'][prop][q]['mainsnak']['datavalue']['value']:
						if js['claims'][prop][q]['mainsnak']['datavalue']['type']=='wikibase-entityid' and isinstance(js['claims'][prop][q]['mainsnak']['datavalue']['value'], dict):
							qlist.append(js['claims'][prop][q]['mainsnak']['datavalue']['value']['id'])
				if len(qlist) > 0:
					prop_dict[prop] = qlist

			q_dict['predicates'] = prop_dict

			base_class_list = []

			if 'claims' in js and 'P31' in js['claims']:
				for i in range(len(js['claims']['P31'])):
					if 'mainsnak' in js['claims']['P31'][i] and 'datavalue' in js['claims']['P31'][i]['mainsnak'] and 'value' in js['claims']['P31'][i]['mainsnak']['datavalue'] and js['claims']['P31'][i]['mainsnak']['datavalue']['type']=='wikibase-entityid' and 'id' in js['claims']['P31'][i]['mainsnak']['datavalue']['value']: 
						base_class = js['claims']['P31'][i]['mainsnak']['datavalue']['value']['id']
						base_class_list.append(base_class)
				q_dict['instance_of'] = base_class_list


			if count % 100000 == 0 :
				print(count)
			
			if count % 6 ==0:
				outer_dict1[ID] = q_dict

			if count % 6 ==1:
				outer_dict2[ID] = q_dict

			if count % 6 ==2:
				outer_dict3[ID] = q_dict

			if count % 6 ==3:
				outer_dict4[ID] = q_dict

			if count % 6 ==4:
				outer_dict5[ID] = q_dict

			if count % 6 ==5:
				outer_dict6[ID] = q_dict


			if ID[0]=='Q':
			 	if 'labels' in js and 'en' in js['labels'] and 'value' in js['labels']['en']:
			 		item_dict[ID] = js['labels']['en']['value']

			if ID[0]=='P':
				if 'labels' in js and 'en' in js['labels'] and 'value' in js['labels']['en']:
					property_dict[ID] = js['labels']['en']['value']
		except Exception as e:
			print('{}'.format(count))
			logging.exception("Something awful happened!")

	with codecs.open(os.path.join(args.out_dir, 'comp_wikidata_1n.json'), 'w', 'utf-8') as f1:
		json.dump(outer_dict1, f1, indent=4)

	with codecs.open(os.path.join(args.out_dir, 'comp_wikidata_2n.json'), 'w', 'utf-8') as f2:
		json.dump(outer_dict2, f2, indent=4)

	with codecs.open(os.path.join(args.out_dir, 'comp_wikidata_3n.json'), 'w', 'utf-8') as f3:
		json.dump(outer_dict3, f3, indent=4)

	with codecs.open(os.path.join(args.out_dir, 'comp_wikidata_4n.json'), 'w', 'utf-8') as f4:
		json.dump(outer_dict4, f4, indent=4)

	with codecs.open(os.path.join(args.out_dir, 'comp_wikidata_5n.json'), 'w', 'utf-8') as f5:
		json.dump(outer_dict5, f5, indent=4)

	with codecs.open(os.path.join(args.out_dir, 'comp_wikidata_6n.json'), 'w', 'utf-8') as f6:
		json.dump(outer_dict6, f6, indent=4)

	with codecs.open(os.path.join(args.out_dir, 'items_wikidata_n.json'), 'w', 'utf-8') as g:
		json.dump(item_dict, g, indent=4)

	with codecs.open(os.path.join(args.out_dir, 'property_wikidata_n.json'), 'w', 'utf-8') as h:
		json.dump(property_dict, h, indent=4)

	inFp.close()