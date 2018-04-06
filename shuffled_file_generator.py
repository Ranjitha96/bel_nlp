# -*- coding: utf-8 -*-
from __future__ import division
import nltk
import re,os
import glob

path="/home/pannaga/bel_project/500N-KPCrowd-v1/CorpusAndCrowdsourcingAnnotations/train/"
candidate_key_files=glob.glob(path+'*-CrowdCountskey')

for file in candidate_key_files:
	fs=file.split('CrowdCountskey')[0]+'Shuffled'
	os.system('shuf -o '+fs+'<'+file)
