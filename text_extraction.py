import pandas as pd
df = pd.read_csv("srNotes.csv", encoding='utf-8')
asr_df = df[df["details.sr_hw_product_erp_family"] == "ASR903"]
asr_df["notesdata.notes"].fillna("", inplace=True)
asr_df = asr_df.groupby('details.sr_number').agg({'details.sr_number':'first', 'notesdata.notes': ' '.join })

notes = asr_df["notesdata.notes"]


#note = "From: no-reply@cisco.comTo: email-in@cisco.comCc: Subject: 683228350RMA #:  88132080Case ID: BKO-222610Part: A903-FAN-E=Notes from RMA BackLog Management: SO# 154129400 allocated ETA 06-NOV-2017 subject to customs clearance. --Added by Agent(Cisco-GSSC-Resolve-Work-BackOrder.CloseMissedBKOs)"
#note = "This is a Routing Decision NoteEPiC: NoRouting Source: BRERouting Type: Resource Group is CO-HARDWARE Legacy Module is COT Load balancerIntelligent Matching: GDPTechnology: HardwareSubTechnology: Hardware failure, Need Replacement (RMA)Problem Code: HARDWARE_FAILURERouting Segment: NoneTerritory Name: Contract Number: Usage Type: Portfolio ID: CPR Country: INCoverage Template: Other: RGWG mapped CSEs possessing tech skill"
##########Preprocessing###########
#notes = notes.replace(['nan', 'None'], '')
#remove emails
import re
c=0
new_notes = []
for note in notes:
    print(c)
    #print(note)
    c+=1
    regex = r"\S*@\S*\s?"
    note = re.sub(regex, '', note, 0)
    #remove from, to, subject, CASE ID and other numbers after #
    stopwords=['From','To','Case ID',':', 'Subject', 'Part']
    for word in stopwords:
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

asr_df["preprocessed_notes"] = new_notes

asr_df.to_csv("ASR_data.csv",encoding='utf-8')

#############More Pre-processing##############
asr_df = pd.read_csv("ASR_data2.csv", encoding='utf-8')
asr_df["preprocessed_notes"] = asr_df["preprocessed_notes"].fillna('')

docs_complete = asr_df["preprocessed_notes"].tolist()
docs_processed=[]
for doc in docs_complete:
    doc = doc.replace('.', ' ')
    doc = doc.replace(';', ' ')
    doc = doc.replace('"', ' ')
    doc = doc.replace('(', ' ')
    doc = doc.replace(')', ' ')
    doc = doc.replace('[', ' ')
    doc = doc.replace(']', ' ')
    doc = doc.replace('+', ' ')
    doc = re.sub(r"==", "", doc, 0)
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

    
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
stop = set(stopwords.words('english'))
stoplist = set('also can thank dear that and other www html com en be have ll will here use make people know many call include part find become like mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask would way something need things want every str please details look note refer to detail with the to is a as am I have been this for . ( )'.split())
lemma = WordNetLemmatizer()  
def clean_text(inputStr):
    stop_free = " ".join([i for i in str(inputStr).lower().split() if i not in (stop and stoplist)])
    normalized = " ".join(lemma.lemmatize(word,'v') for word in stop_free.split())
    x = normalized.split()
    #y = [s for s in x if len(s) > 2]
    #return ' '.join(x)
    return x

###########Any cleaning to sentences needs to be done here itself###############
c=0
final_docs=[]
for doc in docs_processed:
    np_extractor = NPExtractor(doc)
    result = np_extractor.extract()
    final_docs.append(" ".join(result))
    print(c)
    c=c+1

docs_clean = [clean_text(doc) for doc in final_docs]
# Build word dictionary
from gensim import corpora
dictionary = corpora.Dictionary(docs_clean)

# Filter terms which occurs in less than 4 articles & more than 40% of the articles 
dictionary.filter_extremes(no_below=4, no_above=0.4)

# List of few words which are removed from dictionary as they are content neutral
stoplist = set('also use make people know many call include part find become like mean often different usually take wikt come give well get since type list say change see refer actually iii aisne kinds pas ask would way something need things want every str please details look note refer to detail with the to is a as am I have been this . ( )'.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]

dictionary.filter_tokens(stop_ids)

# Feature extraction - Bag of words
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs_clean]

# Creating the object for LDA model using gensim library & Training LDA model on the document term matrix.
from gensim.models.ldamodel import LdaModel as Lda
from gensim.models.hdpmodel import HdpModel as Hdp
ldamodel = Lda(doc_term_matrix, num_topics=50, id2word = dictionary)#, passes=50, iterations=500)
hdpmodel = Hdp(doc_term_matrix, id2word=dictionary)

# Print all the 50 topics
for i,topic in enumerate(ldamodel.print_topics(num_topics=50, num_words=10)):
   words = topic[1].split("+")
   print (words)

# dump LDA model using cPickle for future use 
ldafile = open('lda_model_3.pkl','wb')
cPickle.dump(ldamodel, ldafile)
hdpfile = open('hdp_model_3.pkl','wb')
cPickle.dump(hdpmodel, hdpfile)
ldafile.close()

#########Run the extraction algorithm#########
import nltk
from nltk.corpus import brown
import sys
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
     (r'.*ed$', 'VBD'),
     (r'.*', 'NN')
])

unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)


cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
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
                matches.append(t[0])
        return matches

key_words = []
c=0
asr_df["preprocessed_notes"].fillna("", inplace=True)
new_notes = asr_df["preprocessed_notes"]
for note in new_notes:
    np_extractor = NPExtractor(note)
    result = np_extractor.extract()
    key_words.append(result)
    print(c)
    c=c+1

asr_df["keywords"] = key_words
asr_df.to_csv("ASR_data2.csv",encoding='utf-8')

asr_df = asr_df.append(csr_df)
final_df=pd.DataFrame()
final_df["sr_number"] = asr_df["details.sr_number"]
#final_df["sr_underlying_cause_desc"] = asr_df["
final_df["sr_resolution_code"] = asr_df["details.sr_resolution_code"]
#final_df["sr_problem_summary"]
final_df["keywords"]= asr_df["keywords"]
final_df["Product_family"] = asr_df["details.sr_hw_product_erp_family"]
import json
records = json.loads(final_df.T.to_json()).values()

import pymongo
username = "csaprw"
passwd = "csaprw123"
hostname = "sjc-wwpl-fas4"
port = "27017"
db = "csap_prd"

mongo_connection_string="mongodb://"+username+":"+passwd+"@"+hostname+":"+port+"/"+db
client=pymongo.MongoClient(mongo_connection_string)
db=client.get_database(db)
db.SR_results.drop()
db.SR_results.insert(records)



########################Modeling##########################
import pandas as pd
asr_df = pd.read_csv("ASR_data2.csv", encoding='utf-8')
asr_df["preprocessed_notes"] = asr_df["preprocessed_notes"].fillna('')
import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint

%matplotlib inline

a = asr_df["preprocessed_notes"].tolist()
def build_texts(a):
    for line in a:
        yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)

train_texts = list(build_texts(a))
dictionary = Dictionary(train_texts)
corpus = [dictionary.doc2bow(text) for text in train_texts]

######HDP######
hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
hdpmodel.show_topics()
topics = hdpmodel.print_topics()
import _pickle as cPickle
hdpfile = open('hdp_model_.pkl','wb')
cPickle.dump(ldamodel,hdpfile)

####Running in the background####
nohup python modeling.py > model_testing.log 2>&1
ps -ef | grep sumreddi
kill -9 PID

##############LDA###############
ldamodel = LdaModel(corpus=corpus, num_topics=len(topics), id2word=dictionary)
ldafile = open('lda_model_.pkl','wb')
cPickle.dump(ldamodel,ldafile)
ldamodel = cPickle.load(open('lda_model_new.pkl', 'rb'))

###########LDA-Mapper###########
#http://christop.club/2014/05/06/using-gensim-for-lda/
doc_topics = []
for doc in doc_term_matrix:
    doc_topics.append(ldamodel.__getitem__(doc, eps=0))

#id2word = gensim.corpora.Dictionary()
#_ = id2word.merge_with(corpus.id2word)
#query = id2word.doc2bow(query)
#print(ldamodel[doc])#[(45, 1), (42, 1), (41, 1), (44, 1)]])
#Create a dataframe and append the topic number to that column
c=0
doc_topics=[]
prob=[]
for doc in doc_term_matrix:
    print(c)#[(45, 1), (42, 1), (41, 1), (44, 1)]])
    a = sorted(ldamodel[doc], key=lambda x: x[1])[-1]
    doc_topics.append(a[0])
    prob.append(a[1])
    c=c+1

asr_df["topic_number"] = doc_topics
asr_df["topic_probability"] = prob

import pymongo
username = "csaprw"
passwd = "csaprw123"
hostname = "sjc-wwpl-fas4"
port = "27017"
db = "csap_prd"

mongo_connection_string="mongodb://"+username+":"+passwd+"@"+hostname+":"+port+"/"+db
client=pymongo.MongoClient(mongo_connection_string)
db=client.get_database(db)

import json
asr_df = asr_df.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1)
asr_df.to_csv("SR_topic_classification.csv", encoding='utf-8')
#records = json.loads(final_test_df.T.to_json()).values()
db.SR_topic_classification.drop()
final_df = pd.DataFrame()
final_df["SR_number"] = asr_df['details.sr_number']
final_df["PF"] = asr_df['details.sr_hw_product_erp_family']
final_df["topic"] = asr_df['topic_number']
final_df["topic_probability"] = asr_df['topic_probability']
records = json.loads(final_df.T.to_json()).values()

db.SR_topic_classification.insert(records)


###########FOR A NEW INPUT############
ldamodel = cPickle.load(open('lda_model_3.pkl', 'rb'))
query = "faulty engineer rma hours crc decision legacy" #6,7
query = query.split()

with open('dictionary.pickle', 'rb') as handle:
    dictionary2 = cPickle.load(handle)

query = dictionary.doc2bow(query)
topics = ldamodel[query]
topic_list = [x[0] for x in topics]