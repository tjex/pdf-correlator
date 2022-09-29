#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding: utf-8

import os, glob, re, io, random, gensim, smart_open, logging, collections
import numpy as np
import pandas as pd

from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as fallback_text_extraction
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from nltk.tokenize import word_tokenize

pdfReaders = []
pdfFiles = []
docLabels = []

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootDir = "/Users/tillman/t-root/dev/projects/2022/pdf-correlator/gitignored"
txtExtractDir = rootDir + '/txt-extractions'
modelDataDir = rootDir + '/model-data'
zoteroDir = '/Users/tillman/t-root/zotero/storage'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# In[2]:


# read files
print("reading pdfs in" + str(rootDir) + " (including subdirectories)")
def read_files():
    os.chdir(rootDir)
    for file in glob.glob("**/*.pdf"):
        try:
            pdfFiles.append(file)
            pdfReaders.append(PdfReader(file))
        except:
            print(bcolors.FAIL + "error: " + file + " is unreadable by glob.glob. Skipping file" + bcolors.ENDC)
    print(bcolors.OKGREEN + "pdf files read" + bcolors.ENDC)
    print()
read_files()


# In[3]:


# extract text from pdfs to designated directory and save as txt files.
def extract_to_txt():
    print("Extracting pdfs to text files (duplicate pdfs are handled automagically)")
    os.chdir(txtExtractDir)
    counter = 0
    text = ""
    for i in pdfReaders:
        counter += 1
        with open(str([i.metadata.title]) + ".txt", 'w', encoding="utf-8") as file:
      
            # add doc title to array for reference / tagging
            docLabels.append(i.metadata.title)
            try:
                for j in range(len(i.pages)):
                    # format txt file so that each document is one one line (doc2vec requirement)
                    text += i.getPage(j).extract_text()
                    text = "".join(line.strip("\n") for line in text)  

                
                    
            except Exception as exc:
                print(bcolors.FAIL + "error: " + "\"" + str(i.metadata.title) + "\"" + " not extractable with PyPDF2, trying with pdfminer" + bcolors.ENDC)
                print()
                # format txt file so that each document is one one line (doc2vec requirement)
                text += fallback_text_extraction(rootDir + "/" + pdfFiles[counter])
                text = "".join(line.strip("\n") for line in text)     
                
 
            file.write(text)
    print(bcolors.OKGREEN + "pdf extraction complete" + bcolors.ENDC)
extract_to_txt()


# In[4]:


# generate a training corpus from all txt files found in designated directory
class CorpusGen(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self, tokens_only=False):
        counter = 0
        for fname in os.listdir(self.dirname):
            
            with smart_open.open(fname, encoding="iso-8859-1") as f:
                for i, line in enumerate(f):
                    tokens = gensim.utils.simple_preprocess(line, min_len=3, max_len=20, deacc=True)
                    if tokens_only:
                        yield tokens
                    else:
                        yield gensim.models.doc2vec.TaggedDocument(tokens, [counter])
            counter += 1
        
trainCorpus = list(CorpusGen('/Users/tillman/t-root/dev/projects/2022/pdf-correlator/gitignored/txt-extractions'))


# In[5]:


# establish a model and build the vocab
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(trainCorpus)
model.train(trainCorpus, total_examples=model.corpus_count, epochs=model.epochs)


# In[11]:


# generate and format data files for tensorboard visualisation
os.chdir(modelDataDir)
model.save_word2vec_format('doc_tensor.w2v', doctag_vec=True, word_vec=False)
python -m gensim.scripts.word2vec2tensor -i doc_tensor.w2v -o pdf_plot

text = ""    
with open('pdf_plot_metadata.tsv', 'w') as file:
    file.write('title\n')
    for fname in os.listdir(txtExtractDir):
        if fname.endswith('.txt'):
            text = fname
            text = re.sub('\[\'', '', text)
            text = re.sub('\'\].txt', '', text)
            text = re.sub('\[', '', text)
            text = re.sub('\].txt', '', text)     
            print(text)
            file.write("%s\n" % text)
        else:
            continue
        


# In[7]:


# word occurence check
checkWord = "the"
print("\"" + str(checkWord) + "\"" + " appears this many times in corpus:")
print({model.wv.get_vecattr(checkWord, 'count')})
print()


# In[8]:


# infer a vector from corupus (I dont actually know (yet) what this means or does! :D )
print("infering a default vector")
print()
vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])


# In[9]:


# save the entire corpus to a txt file
with open(modelDataDir + "/" + "train-corpus.txt", 'w') as file:
    file.write(str(trainCorpus))


# In[12]:


# assessing the model
print("assessing the model (this can take a while)")
ranks = []
secondRanks = []
for doc_id in range(len(trainCorpus)):
        inferredVector = model.infer_vector(trainCorpus[doc_id].words)
        sims = model.dv.most_similar([inferredVector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        secondRanks.append(sims[1])

counter = collections.Counter(ranks)
print(bcolors.OKGREEN + "model assessed, all is well in computer land" + bcolors.ENDC)


# In[ ]:





# In[ ]:




