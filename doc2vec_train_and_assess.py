# coding: utf-8

import os, glob, re, io, random, gensim, smart_open, logging, collections
import numpy as np

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
txtExtractDir = "/Users/tillman/t-root/dev/projects/2022/pdf-correlator/gitignored/txt-extractions/"
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

# extract text from pdfs to designated directory and save as txt files.
def extract_to_txt():
    os.chdir(txtExtractDir)
    pat0 = ('(?<!Dr)(?<!Esq)\. +(?=[A-Z])')
    pat1 = ('\.+(?=[A-Z])')
    pat2 = ('\.+(?=[0-9])')
    pat3 = ('\. +(?=[0-9])')
    pat4 = ('(?=[for a of the and to in])')
    
    patterns = [pat0, pat1, pat2, pat3, pat4]
    counter = 0
    text = ""
    for i in pdfReaders:
        counter += 1
        with open(str([i.metadata.title]) + ".txt", 'w', encoding="utf-8") as file:
            
            # add doc title to array for reference / tagging
            docLabels.append(i.metadata.title)
            print("excracting: " + str(i.metadata.title))
            print()
            try:
                for j in range(len(i.pages)):
                    # format txt file so that each line is one sentence (doc2vec requirement)
                    text += i.getPage(j).extract_text()
                    text = re.sub(patterns[0], '.\n', text)
                    text = re.sub(patterns[1], '.\n', text)
                    text = re.sub(patterns[2], '.\n', text)
                    text = re.sub(patterns[3], '.\n', text)
                    text = re.sub(patterns[4], '', text)
                  
                    
            except Exception as exc:
                print(bcolors.FAIL + "error: " +  "pdf not extractable with PyPDF2, trying with pdfminer" + bcolors.ENDC)
                print()
                # format txt file so that each line is one sentence (doc2vec requirement)
                text += fallback_text_extraction(rootDir + "/" + pdfFiles[counter])
                text = re.sub(patterns[0], '.\n', text)
                text = re.sub(patterns[1], '.\n', text)
                text = re.sub(patterns[2], '.\n', text)
                text = re.sub(patterns[3], '.\n', text)
                text = re.sub(patterns[4], '', text)
            file.write(text)   

extract_to_txt()

# generate a training corpus from all txt files found in designated directory 
class CorpusGen(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self, tokens_only=False):
        for fname in os.listdir(self.dirname):
            with smart_open.open(fname, encoding="iso-8859-1") as f:
                for i, line in enumerate(f):
                    tokens = gensim.utils.simple_preprocess(line, min_len=3, max_len=15, deacc=True)
                    if tokens_only:
                        yield tokens
                    else:
                        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

trainCorpus = list(CorpusGen('/Users/tillman/t-root/dev/projects/2022/pdf-correlator/gitignored/txt-extractions'))


# establish a model and build the vocab
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, epochs=40)
model.build_vocab(trainCorpus)

# word occurence check 
checkWord = "the"
print(str(checkWord) + " appears this many times in corpus:")
print({model.wv.get_vecattr(checkWord, 'count')})
print()
model.train(trainCorpus, total_examples=model.corpus_count, epochs=model.epochs)


# infer a vector from corupus (I dont actually know (yet) what this means or does! :D )
print("infering a default vector")
print()
vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])

# save the entire corpus to a txt file
with open(txtExtractDir + "traincorpus.txt", 'w') as file:
    file.write(str(trainCorpus))

# assessing the model
print("assessing the model (this can take a while)")
print()
ranks = []
secondRanks = []
for doc_id in range(len(model.dv)):
        inferredVector = model.infer_vector(trainCorpus[doc_id].words)
        sims = model.dv.most_similar([inferredVector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        secondRanks.append(sims[1])

counter = collections.Counter(ranks)


print('Document ({}): «{}»\n'.format(doc_id, ' '.join(trainCorpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(trainCorpus[sims[index][0]].words)))

print()
print(bcolors.OKGREEN + "doc2vec training and assessment successful" + bcolors.ENDC)
