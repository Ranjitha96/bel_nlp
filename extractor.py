# -*- coding: utf-8 -*-
import os,shutil,re
from bs4 import BeautifulSoup
src='/home/pannaga/bel project/corpus/citations_class'
allfiles=os.listdir(src)
dest='/home/pannaga/bel project/corpus/train_data/'
trainfiles=allfiles[-1927:]
for i in trainfiles:
	st = i.split('.')[0]
	st=st+'.txt'
	file=open(os.path.join(src,i),'r')
	dest_file=open(os.path.join(dest+st),'w')
	raw=file.read().decode('ISO-8859-1')
	soup=BeautifulSoup(raw,"lxml")
	html_free=soup.get_text(strip=True)	
	processed=re.sub('[^A-Za-z .-]+',' ', html_free)
	processed=processed.replace('-',' ').replace('.',' ').replace('...',' ')
	dest_file.write(processed)
