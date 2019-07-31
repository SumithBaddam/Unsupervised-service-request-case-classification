#########Modeling the SR data for underlying cause desc#############
import pandas as pd
import re
import nltk
from nltk.corpus import brown
import sys
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel as Lda
import pickle
import _pickle as cPickle
import json
import pymongo
import os
import configparser
import sys
import shutil
import argparse
import numpy as np
import jsondatetime as json2
sys.path.insert(0, "/data/ingestion/")
from Utils import *

#Parser options
options = None

def parse_options():
    parser = argparse.ArgumentParser(description="""Script to classify SR cases for underlying cause code""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    args = parser.parse_args()   
    return args

brown_train = brown.tagged_sents(categories='news')
regexp_tagger = nltk.RegexpTagger(
    [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
     (r'(-|:|;)$', ':'),
     (r'\'*$', 'MD'),
     (r'(The|the|A|a|An|an)$', 'AT'),
     (r'.*able$', 'JJ'),
     (r'^[A-Z].*$', 'NNP'),
     (r'.*ness$', 'NN'),
     (r'.*ly$', 'RB'),
     (r'.*s$', 'NNS'),
     (r'.*ing$', 'VBG'),
     (r'.*n t$', 'VBG'),
     (r'.*ed$', 'VBD'),
     (r'.*', 'NN')
])

unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)

cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
#cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"


class NPExtractor(object):
    def __init__(self, sentence):
        self.sentence = sentence
    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens
    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged
    def extract(self):
        tokens = self.tokenize_sentence(self.sentence)
        tags = self.normalize_tags(bigram_tagger.tag(tokens))
        #print tags
        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = cfg.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = "%s %s" % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break
        matches = []
        for t in tags:
            #if t[1] == "NNP" or t[1] == "NNI":
            if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":
                #if(t[1] != "NNI"):
                    #print(t)
                matches.append(t[0])
        return matches



stop = set(stopwords.words('english'))
stoplist = set('SR also use make people know many call include part find become like mean often different usually take wikt come give well get since type list say sr also asa can thank #name dear email day when session por port txt that and other www html name com en be have ll will here use make people know many call include part find become like | mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask also use make people know many call include part find become like mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask would way something need things want every str please crs asr asa done cisco use asa make com cause sr team html cause web save subject sure thanks regards ensure work also detail case reply sr function https browser windows mozilla chrome safari link lead user use type important to from update log all apac active contact status attachment index web html free regardto cisco http mycase subject sincerely ist january february haven possible aim tmr helpdesk fine save load support note description done response timely query request hours yet regard service notification device kindly hi hey good email manager employee id test option time please details file attachment thanks regards state program ok look no yes type insert asr903 detailasr903 can thank mac macintosh when #name dear that and other www html name com en be have ll will here use make people know many call include part find become like | mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask would way something need things want every str please details look note refer to detail with the to is a as am I have been this for . ( )'.split())
lemma = WordNetLemmatizer()
def clean_text(inputStr):
    stop_free = " ".join([i for i in str(inputStr).lower().split() if i not in (stop and stoplist)])
    normalized = " ".join(lemma.lemmatize(word,'v') for word in stop_free.split())
    x = normalized.split()
    y = [s for s in x if len(s) > 2]
    #return ' '.join(x)
    return y

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')
collection_prefix = 'SRNotes_' #str(settings.get("SR_Source","srFilesLoc"))
model_path = '/auto/vgapps-cstg02-vapps/analytics/csap/models/files/sr/' #'/data/csap_models/srData/'

#Database setup
options = parse_options()
if(options.env.lower() == "prod"):
	key = "csap_prod_database"
elif(options.env.lower() == "stage"):
	key = "csap_stage_database"

db = get_db(settings, key)

#For each PF, run the function
pf_list = settings.get("SR_Source","pfList").split(',')
pf_list = ['ASR903', 'CRS', 'ASA', 'ASR9000', 'ASR901', 'ASR920', 'N9K ACI', 'N9K Standalone', 'NCS5500', 'Tetration', 'UCSB', 'UCSC', 'UCSHX', 'Ultra', 'VPCSW', 'White Box'] #['ASA', 'ASR903', 'CRS']

print(pf_list)
if(len(pf_list) > 0):
	db.SR_topic_keywords_cause_desc.drop()
	db.SR_keywords_SRcases_cause_desc.drop()
	print("Dropped the 2 collections")

for pf in pf_list:
	collection = db[collection_prefix + str(pf)]
	print(collection)
	cursor = collection.find({})
	#asa_df = pd.read_csv(filepath + "srNotes_" + pf + ".csv", encoding='utf-8')
	asa_df = pd.DataFrame(list(cursor))

	#Write try exception block

	asa_df["notes"].fillna("", inplace=True)
	asa_df['sr_resolution_code'].fillna("", inplace=True)
	asa_df['sr_underlying_cause_code'].fillna("", inplace=True)
	asa_df['sr_troubleshooting_description'].fillna("", inplace=True)
	asa_df['sr_problem_summary'].fillna("", inplace=True)
	asa_df['sr_underlying_cause_desc'].fillna("", inplace=True)
	notes = asa_df["notes"]+' '+asa_df['sr_resolution_code']+' '+asa_df['sr_underlying_cause_code']+' '+asa_df['sr_troubleshooting_description']+' '+asa_df['sr_problem_summary'] + asa_df['sr_underlying_cause_desc']
	notes = notes.replace(np.nan, '', regex=True)

	##########Preprocessing###########
	c=0
	new_notes = []
	for note in notes:
		#print(c)
		#print(note)
		c+=1
		regex = r"\S*@\S*\s?"
		note = re.sub(regex, '', note, 0)
		#remove from, to, subject, CASE ID and other numbers after #
		stop_words=['From','To','Case ID',':', 'Subject', 'Part']
		for word in stop_words:
			if word in note:
				note=note.replace(word,"")
		note = re.sub(' +',' ',note)
		regex = r"\S* # \S*\s?"
		note = re.sub(regex, '', note, 0)
		regex = r"\S*# \S*\s?"
		note = re.sub(regex, '', note, 0)
		#remove dates and time
		regex = r"\d\d-\S*-\d\d\d\d*"
		note = re.sub(regex, '', note, 0)
		note = re.sub(' +',' ',note)
		new_notes.append(note)

	asa_df["preprocessed_notes"] = new_notes
	asa_df["preprocessed_notes"] = asa_df["preprocessed_notes"].fillna('')

	docs_complete = asa_df["preprocessed_notes"].tolist()
	docs_processed=[]
	for doc in docs_complete:
		doc = doc.replace('*', ' ')
		doc = doc.replace('=', ' ')
		doc = doc.replace('__', ' ')
		doc = doc.replace('.', ' ')
		doc = doc.replace(';', ' ')
		doc = doc.replace('"', ' ')
		doc = doc.replace('(', ' ')
		doc = doc.replace(')', ' ')
		doc = doc.replace('[', ' ')
		doc = doc.replace(']', ' ')
		doc = doc.replace('+', ' ')
		doc = doc.replace('|', ' ')
		doc = doc.replace('#', ' ')
		doc = re.sub(r"==", " ", doc, 0)
		doc = re.sub(r"-", " ", doc, 0)
		doc = re.sub(r"&", " ", doc, 0)
		doc = re.sub(r"--", "", doc, 0)
		doc = re.sub(r"{", " ", doc, 0)
		doc = re.sub(r"}", " ", doc, 0)
		doc = re.sub(r":", " ", doc, 0)
		doc = re.sub(r"/", " ", doc, 0)
		doc = re.sub(r">", " ", doc, 0)
		doc = re.sub(r"<", " ", doc, 0)
		doc = re.sub(r",", " ", doc, 0)
		doc = re.sub(r"'", " ", doc, 0)
		doc = re.sub(r"!", " ", doc, 0)
		doc = re.sub(r"@", " ", doc, 0)
		doc = re.sub(r"GMT", " ", doc, 0)
		docs_processed.append(doc)

	print("Completed initial preprocessing")
	c=0
	final_docs_asa=[]
	for doc in docs_processed:
		np_extractor = NPExtractor(doc)
		result = np_extractor.extract()
		final_docs_asa.append(" ".join(result))
		#print(c, len(docs_processed))
		c=c+1
	
	print("Completed keyword extraction")
	asa_df["final_notes"] = final_docs_asa
	#asa_df.to_csv(filepath + "srNotes_ASA_processed.csv", encoding='utf-8')

	asa_df = asa_df.groupby('sr_number').agg({'sr_number':'first', 'final_notes': ' '.join, 'sr_hw_product_erp_family':'first', 'sr_create_timestamp':'first', 'sr_defect_number':'first', 'sr_resolution_code':'first', 'sr_underlying_cause_code': 'first', 'sr_underlying_cause_desc': 'first', 'sr_problem_summary': 'first', 'sr_troubleshooting_description': 'first'})

	b = 0
	final_key_df = pd.DataFrame()
	final_topic_class_df = pd.DataFrame()
	final_keyword_srcases_df = pd.DataFrame()
	cause_descs = list(asa_df['sr_underlying_cause_desc'].unique())
	cause_codes = list(asa_df['sr_underlying_cause_code'].unique())
	print("Running on cause_descs")
	#for desc in cause_descs:
	for i in range(0, len(cause_descs)):
		desc = cause_descs[i]
		ccode = cause_codes[i]
		df = asa_df[asa_df['sr_underlying_cause_desc'] == desc]
		#print(df.shape)
		docs_clean = [clean_text(doc) for doc in df["final_notes"]]
		# Build word dictionary
		dictionary = corpora.Dictionary(docs_clean)
		# List of few words which are removed from dictionary as they are content neutral
		stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
		dictionary.filter_tokens(stop_ids)
		#print(dictionary.token2id)
		# Feature extraction - Bag of words
		# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
		filename = model_path + pf + '_' + ccode + '_dictionary.pickle' #/auto/vgapps-cstg02-vapps/analytics/csap/models/files/sr/
		if os.path.exists(filename):
			os.remove(filename)
			print("File Removed!")

		with open(filename, 'wb') as handle:
			pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

		doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs_clean]
		if(len(doc_term_matrix) > 0):
			# Creating the object for LDA model using gensim library & Training LDA model on the document term matrix.
			num_topics = int(len(doc_term_matrix)/20)
			if(num_topics < 2):
				num_topics = 2
			if(num_topics > 30):
				num_topics = 30
			#ldamodel = Lda(doc_term_matrix, num_topics = num_topics, id2word = dictionary)#, passes=50, iterations=500)
			ldamodel = Lda(doc_term_matrix, num_topics = num_topics, id2word = dictionary, passes=3)
			print("Saving the LDA model")

			filename = model_path + pf + '_' + ccode + '_lda_model.pkl' #/auto/vgapps-cstg02-vapps/analytics/csap/models/files/sr/
			if os.path.exists(filename):
				os.remove(filename)
				print("File Removed!")

			ldafile = open(filename,'wb')
			cPickle.dump(ldamodel, ldafile)
			print("File Created!")
			ldafile.close()

			topics_words = ldamodel.print_topics(num_topics = num_topics, num_words = 10)
			c=0
			doc_topics=[]
			prob=[]
			for doc in doc_term_matrix:
				a = sorted(ldamodel[doc], key=lambda x: x[1])[-1]
				doc_topics.append(a[0])
				prob.append(a[1])
				c=c+1
			df["topic_number"] = doc_topics
			df["topic_probability"] = prob
			#Create a dataframe and append the topic number to that column
			final_df = pd.DataFrame()
			final_df["SR_number"] = df['sr_number']
			final_df["PF"] = pf #asa_df['sr_hw_product_erp_family']
			final_df["topic"] = df['topic_number']
			final_df["topic_probability"] = df['topic_probability']
			final_topic_class_df = final_topic_class_df.append(final_df)
			topics_matrix = ldamodel.show_topics(formatted=False,num_words=10, num_topics = num_topics)
			topics_matrix = np.array((topics_matrix),dtype=list)
			topics_df = pd.DataFrame()
			top_probs = []
			top_words = []
			top = []
			for topic in range(0, num_topics):
				a = topics_matrix[topic]
				for i in range(0,10):
					top.append(topic)
					top_words.append(a[1][i][0])
					top_probs.append(a[1][i][1])
			topics_df['Topic_number'] = top
			topics_df['keyword'] = top_words
			topics_df['probability'] = top_probs
			topics_df['PF'] = pf
			topics_df['underlying_cause_desc'] = desc
			topics_df['underlying_cause_code'] = ccode
			#final_key_df = final_key_df.append(topics_df)

			sample_1 = topics_df[['keyword', 'probability', 'PF', 'underlying_cause_desc', 'underlying_cause_code']]
			sample_1.reset_index(drop = True, inplace = True)
			records = json2.loads(sample_1.T.to_json(date_format='iso')).values()
			db.SR_topic_keywords_cause_desc.insert(records)

			print(desc)
			tk_asa_df = topics_df
			asa_df_1 = final_df
			list_of_keywords_asa = top_words
			keyword_srcases_df = pd.DataFrame()
			for word in list_of_keywords_asa:
				new_word = lemma.lemmatize(word,'v')
				sr_cases = []
				temp = tk_asa_df[tk_asa_df['keyword'] == new_word]
				if(temp.size != 0):
					topics = temp.Topic_number.unique()
					for topic in topics:
						temp2 = asa_df_1[asa_df_1['topic']==topic][asa_df_1['topic_probability'] > 0.87]
						sr_cases = sr_cases + list(temp2.SR_number.unique())
					b = b + 1
					df3 = pd.DataFrame([[new_word, sr_cases, len(sr_cases), b, pf, desc, ccode]], columns=['Keyword', 'SR_number', 'Count_of_cases', 'ID', 'PF', 'underlying_cause_desc', 'underlying_cause_code'])
					keyword_srcases_df = keyword_srcases_df.append(df3)
			#final_keyword_srcases_df = final_keyword_srcases_df.append(keyword_srcases_df)

			sample_2 = keyword_srcases_df.set_index(['ID'])
			records = json.loads(sample_2.T.to_json()).values()
			db.SR_keywords_SRcases_cause_desc.insert(records)
	'''
	final_key_df = final_key_df[['keyword', 'probability', 'PF', 'underlying_cause_desc']]
	final_key_df.reset_index(drop = True, inplace = True)
	records = json2.loads(final_key_df.T.to_json(date_format='iso')).values()
	#db.SR_topic_keywords_cause_desc.drop()
	db.SR_topic_keywords_cause_desc.insert(records)

	final_keyword_srcases_df = final_keyword_srcases_df.set_index(['ID'])
	records = json.loads(final_keyword_srcases_df.T.to_json()).values()
	#db.SR_keywords_SRcases_cause_desc.drop()
	db.SR_keywords_SRcases_cause_desc.insert(records)
	'''