#########Generating report for SR analytics#############
import pandas as pd
import re
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
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')

def parse_options():
    parser = argparse.ArgumentParser(description="""Script to classify SR cases for underlying cause code""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    args = parser.parse_args()   
    return args

def get_top_keywords(df, pf):
	df = df[df['PF'] == pf]
	result = df.sort_values(['Count_of_cases'], ascending=[0])
	result = result[result['Count_of_cases'] > 0]
	result = result.drop_duplicates('Keyword')
	return result

options = parse_options()
if(options.env.lower() == "prod"):
	key = "csap_prod_database"
elif(options.env.lower() == "stage"):
	key = "csap_stage_database"

db = get_db(settings, key)

pf_list = settings.get("SR_Source","pfList").split(',')
pf_list = ['ASR903', 'CRS', 'ASA', 'ASR9000', 'ASR901', 'ASR920', 'N9K ACI', 'N9K Standalone', 'NCS5500', 'Tetration', 'UCSB', 'UCSC', 'UCSHX', 'Ultra', 'VPCSW', 'White Box'] #['ASA', 'ASR903', 'CRS']

collection = db.SR_keywords_SRcases_cause_desc
print(collection)
cursor = collection.find({})
df = pd.DataFrame(list(cursor))
keywords_len = 30

final_df = pd.DataFrame()
for pf in pf_list:
	print(pf)
	df_keywords = get_top_keywords(df, pf)
	print(df_keywords.PF.unique())
	final_df = final_df.append(df_keywords)

final_df.to_csv('SR_analysis.csv', encoding = 'utf-8')