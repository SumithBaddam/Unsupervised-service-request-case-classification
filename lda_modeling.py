# python ./lda_modeling.py --env prod
#########Modeling the SR data#############
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
import numpy as np
import configparser
import sys
import shutil
import argparse
sys.path.insert(0, "/data/ingestion/")
from Utils import *

#Parser options
options = None

def parse_options():
    parser = argparse.ArgumentParser(description="""Script to classify SR cases""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    args = parser.parse_args()   
    return args

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')
collection_prefix = 'SRNotes_' #filepath = str(settings.get("SR_Source","srFilesLoc"))
model_path = '/data/csap_models/srData/' #'/auto/vgapps-cstg02-vapps/analytics/csap/models/files/sr/'

#####Database configuration#####
options = parse_options()
if(options.env.lower() == "prod"):
    key = "csap_prod_database"
elif(options.env.lower() == "stage"):
    key = "csap_stage_database"

db = get_db(settings, key)
print(db)

######NLP parameters config######
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


######Preprocessing of data######
stop = set(stopwords.words('english'))
#stoplist = set('done ensure work also detail case reply sr function https browser windows mozilla chrome safari link lead user use type important to from update log all apac active contact status attachment index web html free regardto cisco http mycase subject sincerely ist january february haven possible aim tmr helpdesk fine save load support note description done response timely query request hours yet regard service notification device kindly hi hey good email manager employee id test option time please details file attachment thanks regards state program ok look no yes type insert asr903 detailasr903 can thank mac macintosh when #name dear that and other www html name com en be have ll will here use make people know many call include part find become like | mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask would way something need things want every str please details look note refer to detail with the to is a as am I have been this for . ( )'.split())
stoplist = set('SR also use make people know many call include part find become like mean often different usually take wikt come give well get since type list say sr also asa can thank #name dear email day when session por port txt that and other www html name com en be have ll will here use make people know many call include part find become like | mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask also use make people know many call include part find become like mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask would way something need things want every str please crs asr asa done cisco use asa make com cause sr team html cause web save subject sure thanks regards ensure work also detail case reply sr function https browser windows mozilla chrome safari link lead user use type important to from update log all apac active contact status attachment index web html free regardto cisco http mycase subject sincerely ist january february haven possible aim tmr helpdesk fine save load support note description done response timely query request hours yet regard service notification device kindly hi hey good email manager employee id test option time please details file attachment thanks regards state program ok look no yes type insert asr903 detailasr903 can thank mac macintosh when #name dear that and other www html name com en be have ll will here use make people know many call include part find become like | mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask would way something need things want every str please details look note refer to detail with the to is a as am I have been this for . ( )'.split())
lemma = WordNetLemmatizer()
def clean_text(inputStr):
    normalized = " ".join(lemma.lemmatize(word,'v') for word in str(inputStr).lower().split())
    stop_free = " ".join([i for i in str(normalized).lower().split() if i not in (stop and stoplist)])
    normalized = " ".join(lemma.lemmatize(word,'v') for word in stop_free.split())
    x = normalized.split()
    y = [s for s in x if len(s) > 2]
    return y


#For each PF, run the function
pf_list = settings.get("SR_Source","pfList").split(',')
#pf_list = ['ASR903', 'ASA', 'CRS']

if(len(pf_list) > 0):
    db.SR_topic_classification.drop()
    db.SR_topic_keywords.drop()
    db.SR_concat_case_topic_keywords.drop()
    db.SR_keywords_SRcases.drop()
	print("Dropped all 4 collections")

for pf in pf_list:
    #asr_df = pd.read_csv(filepath + "srNotes_" + pf + ".csv", encoding='utf-8')
    collection = db[collection_prefix + str(pf)]
	print(collection)
	cursor = collection.find({})
	asa_df = pd.DataFrame(list(cursor))

	asr_df["notes"].fillna("", inplace=True)
    asr_df['sr_resolution_code'].fillna("", inplace=True)
    asr_df['sr_underlying_cause_code'].fillna("", inplace=True)
    asr_df['sr_troubleshooting_description'].fillna("", inplace=True)
    asr_df['sr_problem_summary'].fillna("", inplace=True)
    asr_df['sr_underlying_cause_desc'].fillna("", inplace=True)

    notes = asr_df["notes"]+' '+ asr_df['sr_resolution_code']+ ' '+ asr_df['sr_underlying_cause_code']+' '+ asr_df['sr_troubleshooting_description']+ ' '+ asr_df['sr_problem_summary'] + asr_df['sr_underlying_cause_desc']
    notes = notes.replace(np.nan, '', regex=True)

    ##########Preprocessing###########
    c=0
    new_notes = []
    for note in notes:
        c+=1
        regex = r"\S*@\S*\s?"
        note = re.sub(regex, '', note, 0)
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
    
    print("Completed intial preprocessing")
    asr_df["preprocessed_notes"] = new_notes

    asr_df["preprocessed_notes"] = asr_df["preprocessed_notes"].fillna('')

    docs_complete = asr_df["preprocessed_notes"].tolist()
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

    c=0
    final_docs=[]
    for doc in docs_processed:
        np_extractor = NPExtractor(doc)
        result = np_extractor.extract()
        final_docs.append(" ".join(result))
        c=c+1

    print("Completed keyword extractions")
    asr_df["final_notes"] = final_docs
    #asr_df.to_csv(filepath + "srNotes_" + pf + "_processed.csv", encoding='utf-8')

    asr_df = asr_df.groupby('sr_number').agg({'sr_number':'first', 'final_notes': ' '.join, 'sr_hw_product_erp_family':'first', 'sr_create_timestamp':'first', 'sr_defect_number':'first', 'sr_resolution_code':'first', 'sr_underlying_cause_code': 'first', 'sr_underlying_cause_desc': 'first', 'sr_problem_summary': 'first', 'sr_troubleshooting_description': 'first'})

    docs_clean = [clean_text(doc) for doc in asr_df["final_notes"]]#final_docs]

    # Build word dictionary
    dictionary = corpora.Dictionary(docs_clean)

    # Filter terms which occurs in less than 4 articles & more than 40% of the articles 
    dictionary.filter_extremes(no_below=4, no_above=0.4)

    # List of few words which are removed from dictionary as they are content neutral
    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]

    dictionary.filter_tokens(stop_ids)
    #print(dictionary.token2id)

    #We need to store the dictionary into a file for new input cases
    filename = model_path + pf + '_dictionary.pickle' #/auto/vgapps-cstg02-vapps/analytics/csap/models/files/sr/
    if os.path.exists(filename):
        os.remove(filename)
        print("File Removed!")

    with open(filename, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Feature extraction - Bag of words
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs_clean]

    # Creating the object for LDA model using gensim library & Training LDA model on the document term matrix.
    print("Running the LDA model")
    ldamodel = Lda(doc_term_matrix, num_topics=50, id2word = dictionary)#, passes=50, iterations=500)
    print("Saving the LDA model")

    filename = model_path + pf + '_lda_model.pkl' #/auto/vgapps-cstg02-vapps/analytics/csap/models/files/sr/
    if os.path.exists(filename):
        os.remove(filename)
        print("File Removed!")

    ldafile = open(filename,'wb')
    cPickle.dump(ldamodel, ldafile)
    print("File Created!")
    ldafile.close()

    topics_words = ldamodel.print_topics(num_topics=50, num_words = 200)
    c=0
    doc_topics=[]
    prob=[]
    for doc in doc_term_matrix:
        a = sorted(ldamodel[doc], key=lambda x: x[1])[-1]
        doc_topics.append(a[0])
        prob.append(a[1])
        c=c+1

    print("DataFrame created")
    asr_df["topic_number"] = doc_topics
    asr_df["topic_probability"] = prob

    #asr_df = asr_df.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1)
    #asr_df.to_csv("SR_topic_classification.csv", encoding='utf-8')

    #Create a dataframe and append the topic number to that column
    final_df = pd.DataFrame()
    final_df["SR_number"] = asr_df['sr_number']
    final_df["PF"] = pf #asr_df['sr_hw_product_erp_family']
    final_df["topic"] = asr_df['topic_number']
    final_df["topic_probability"] = asr_df['topic_probability']

    records = json.loads(final_df.T.to_json()).values()
    print("Ran successfully")
    #db.SR_topic_classification.drop()
    db.SR_topic_classification.insert(records)
    print("Created the collection")
    ####################Running the topics and words collection#####################

    topics_matrix = ldamodel.show_topics(formatted=False,num_words=200, num_topics=50)
    topics_matrix = np.array((topics_matrix),dtype=list)
    topics_df = pd.DataFrame()
    top_probs = []
    top_words = []
    top = []
    for topic in range(0, 50):
        a = topics_matrix[topic]
        for i in range(0,200):
            top.append(topic)
            top_words.append(a[1][i][0])
            top_probs.append(a[1][i][1])

    topics_df['Topic_number'] = top
    topics_df['keyword'] = top_words
    topics_df['probability'] = top_probs
    topics_df['PF'] = pf
    
    print("Running topic and keywords")
    records = json.loads(topics_df.T.to_json()).values()
    #db.SR_topic_keywords.drop()
    db.SR_topic_keywords.insert(records)
    print("Completed topic and keywords")

    ####################Running for concatenation of SR number, topic, keywords, probability#####################
    topics_column = final_df['topic'].tolist()
    all_top_words = []
    for t in topics_column:
        a = topics_matrix[t]
        top_words = "" #top_words=[]
        for i in range(0,20):
            top_words = top_words + ',' + str(a[1][i][0])
        all_top_words.append(top_words)    

    asr_df['keywords'] = all_top_words
    final_df1 = pd.DataFrame()
    final_df1['SR_number'] = asr_df['sr_number']
    final_df1['resolution_code'] = asr_df['sr_resolution_code']
    final_df1['underlying_cause_code'] = asr_df['sr_underlying_cause_code']
    final_df1['underlying_cause_desc'] = asr_df['sr_underlying_cause_desc']
    final_df1['description'] = asr_df['sr_troubleshooting_description']
    final_df1['processed_notes'] = asr_df['final_notes']
    final_df1['problem_summary'] = asr_df['sr_problem_summary']
    final_df1['defect_number'] = asr_df['sr_defect_number']
    final_df1['create_timestamp'] = asr_df['sr_create_timestamp']
    final_df1['topic_number'] = asr_df['topic_number']
    final_df1['keywords'] = asr_df['keywords']
    final_df1['topic_probability'] = asr_df['topic_probability']
    final_df1['PF'] = pf

    print("Running Sr, topic an keyword concat")
    records = json.loads(final_df1.T.to_json()).values()
    #db.SR_concat_case_topic_keywords.drop()
    db.SR_concat_case_topic_keywords.insert(records)
    print("Completed Sr, topic an keyword concat")


#########################Mapping of keyword list to SR cases#########################

collection = db['SR_topic_keywords']
cursor = collection.find({}) # query
df1 =  pd.DataFrame(list(cursor))

collection = db['SR_topic_classification']
cursor = collection.find({})
df2 =  pd.DataFrame(list(cursor))
df = pd.DataFrame()

list_of_keywords_asa = ['customer_education','session platform','management','device','useragent','traffic','sw_config','config_assistance','applewebkit','firewall','cri','system','feature','collaboration','audio','review','upgrade','password','client','document','response','csone','troubleshoot','report','address','connection','normal','tunnel','bug','firepower','remote','command','config','materials','anyconnect','sm1','base','svc','comm004','interface','output','reboot','htts','license','mobility','internet','ref_00da0h3hn','comm003','certificate','cws','internal','dcss','tcp','speed','servicedesk','servicedeskpage','salesforce','devices','entitlement','packet','port','asasm','sourcefire','docs','cway','setup','agent','wgs','ssh','run','amm','documentation','info','research','subnet','pin','pst','nat','communication','cloud','fix','switch','policy','modules','data','serial','configure','failover','ips','customers','inspection','context','sfr','csc','cli','pak','image','release','operations','sw_upgrd_exstng_defect','comm009','sntp','tsd_cisco_worldwide_contacts','analysis','asp','net','admin','auto','virtual','scansafe','wan','rma','content','cpu','tracer','application','authentication','entitle','monitor','knowledge','connectivity','mtid','arp','distribution','confidential','packets','autorun','interfaces','trace','fmc','fwsm','portal','deployment','interoperability','smart','translation','database','gce','process','core','signature','software_failure','registration','wsa','lan','exception','interop','asa1000v','hardware','timeout','workaround','cat6500','lab','analyzer','quality','fail','requirements','settings','control','software_assistance','maintenance','gateway','show_tech','advise','gci','agreement','collaborate','Crypto']
list_of_keywords_asr = ['bfd', 'mpls', 'routing', 'issu', 'fan', 'timing', 'tdm', 'qos', 'multicast', 'avail', 'layer2', 'oam', 'snmp', 'im', 'optics', 'availability', 'license', 'rsp', 'bootflash', 'psu', 'alarm', 'memory', 'boot']
list_of_keywords_crs = ['bfd', 'bundle', 'slow', 'ucode', 'lpts', 'feature', 'multicast', 'stat', 'chassis', 'infra', 'cpu', 'fabric', 'npu', 'satellite', 'resource', 'manager', 'rdm', 'edrm', 'uidm']

keyword_list = [list_of_keywords_asa, list_of_keywords_asr, list_of_keywords_crs]

for i in range(len(pf_list)):
	pf = pf_list[i]
	tk_asr_df = df1[df1['PF'] == pf]
	asr_df = df2[df2['PF'] == pf]
	#df = pd.DataFrame()
	#c=0
	for word in keyword_list[i]:
		new_word = lemma.lemmatize(word,'v')
		sr_cases = []
		print(new_word)
		temp = tk_asr_df[tk_asr_df['keyword'] == new_word]
		if(temp.size != 0):
			topics = temp.Topic_number.unique()
			for topic in topics:
				temp2 = asr_df[asr_df['topic']==topic][asr_df['topic_probability'] > 0.75]
				sr_cases = sr_cases + list(temp2.SR_number.unique())
			c = c+1
			df3 = pd.DataFrame([[new_word, sr_cases, len(sr_cases), c, pf]], columns=['Keyword', 'SR_number', 'Count_of_cases', 'ID', 'PF'])
			df = df.append(df3)

d = df.set_index(['ID'])
print('Inserting keyword records to the collection')
records = json.loads(d.T.to_json()).values()
#db.SR_keywords_SRcases.drop()
db.SR_keywords_SRcases.insert(records)
print('Records inserted')