# note - at time of writing, the word2vec2tensor.py script that is run in this code only works in Jupyter notebook
# therefore, if you would like to run the code, and this text is still here to be read, then it means I didn't fix it in time and you will need to run it from jupyter notebook...
# you will also of course need to install the relevant packages (see readme)

# pdf-correlator by Tillman Jex
# github.com/tjex
# tillmanjex@mailbox.org

import os, glob, re, io, random, gensim, smart_open, logging, collections, sys

from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extraction
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

pdfReaders = []
pdfFiles = []
docLabels = []

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootDir = sys.path[0]
pdfReadRootDir = rootDir + "/pdf-tests" # change this to where you would like to search for pdfs from
txtExtractDir = rootDir + '/txt-extractions'
modelDataDir = rootDir + '/model-data'
testsDir = rootDir + '/tests'


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

###### PART 1 - READ, EXTRACT, TRAIN AND ASSESS ######

# read files
print("reading pdfs in" + str(rootDir) + " (including subdirectories)")
def read_files():
    os.chdir(pdfReadRootDir)
    for file in glob.glob("**/*.pdf", recursive=True):
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
                text += pdfminer_extraction(rootDir + "/" + pdfFiles[counter])
                text = "".join(line.strip("\n") for line in text)     
                
 
            file.write(text)
    print(bcolors.OKGREEN + "pdf extraction complete" + bcolors.ENDC)
extract_to_txt()

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
        
trainCorpus = list(CorpusGen(txtExtractDir))



# save the entire corpus to a txt file
with open(modelDataDir + "/train-corpus.txt", 'w') as file:
    file.write(str(trainCorpus))



# establish a model and build the vocab
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(trainCorpus)
model.train(trainCorpus, total_examples=model.corpus_count, epochs=model.epochs)



# generate and format data files for tensorboard visualisation
os.chdir(modelDataDir)
model.save_word2vec_format('doc_tensor.w2v', doctag_vec=True, word_vec=False)
get_ipython().run_line_magic('run', '../scripts/word2vec2tensor.py -i doc_tensor.w2v -o pdf_plot')

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


# word occurence check
checkWord = "the"
print("\"" + str(checkWord) + "\"" + " appears this many times in corpus:") 
print({model.wv.get_vecattr(checkWord, 'count')})
print()

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
print(bcolors.OKGREEN + "model assessed" + bcolors.ENDC)
print()
print(bcolors.OKBLUE + "model trained and assessed successfully" + bcolors.ENDC)


###### PART 2 - CHECK SIMILARITY BETWEEN CORPUS AND INCOMING DOCUMENT ######

# import new document
print('importing latin pdf')
with smart_open.open(testsDir + '/similarity-test.txt', 'w') as test:
    text = pdfminer_extraction(testsDir + '/german.pdf')
    text = "".join(line.strip("\n") for line in text) 
    test.write(text)
    
print()

# tokenize and tag new document
print('tag and tokenize pdf for doc2vec processing')
def read_text(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
                
similarityTest = list(read_text(testsDir + '/similarity-test.txt'))

print()

# check for similarity against the entire corpus
print('compare similarity between incoming pdf and text corpus (the model)')
novel_vector = model.infer_vector(similarityTest[0].words)
similarity_scores = model.dv.most_similar([novel_vector])
average = 0
for score in similarity_scores:
    average += score[1]
overall_similarity = average/len(similarity_scores)

print("> Incoming pdf is " + format((overall_similarity - 1) * -1, '.0%') + " similar to corpus")

