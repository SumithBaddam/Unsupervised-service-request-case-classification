import pandas as pd
import re
import pymongo
import os
import numpy as np

username = "csaprw"
passwd = "csaprw123"
hostname = "sjc-wwpl-fas4"
port = "27017"
db = "csap_prd"

mongo_connection_string="mongodb://"+username+":"+passwd+"@"+hostname+":"+port+"/"+db
client=pymongo.MongoClient(mongo_connection_string)
db=client.get_database(db)
collection = db["SR_topic_keywords"]
cursor = collection.find({}) # query
topics_keywords =  pd.DataFrame(list(cursor))

topics_list = topics_keywords.Topic_number.unique()
pf_list = topics_keywords.PF.unique()

m = 200 # top 10 words from each topic
top_keywords = []

#for pf in pf_list:
#    df = topics_keywords[topics_keywords['PF'] == pf]
for topic in topics_list:
    print(topic)
    df2 = df[df['Topic_number'] == topic]
    #top_keywords = df2[0:m]['keyword'].values.tolist()
    top_keywords.extend(df2[0:m]['keyword'].values.tolist())

no = [x for x in top_keywords if not any(c.isdigit() for c in x)]
no = [x for x in no if not any(ord(c)>126 or ord(c)<63 for c in x)]
no.index('multicast')