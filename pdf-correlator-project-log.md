# Second Term Project Log

### 2022-09-17
Based on the previous experience with the CC2 project I want to spend some solid time making sure I find the right tools / packages / libraires to do the job, before starting. I think it's more time efficient - considering I don't have much of an idea where to start with this project.

Program task breakdown:
    - source current local papers (zotero library / pdfs)
    	- for every .pdf found in directory, create and assign PdfReader object.
    - extract text
    	- write to file
    - train model based on that text
	- clean text
	- ...
    - ascertain patterns / similarities  
    - feed in a new selection of papers
    - ascertain patterns / similarities to the original source papers
    - provide a score - strongly correlated / un correlated to current library of papers

### 2002-09-18
[python tools to extract text from pdfs / read pdfs](https://www.delftstack.com/howto/python/read-pdf-in-python/)

From my understanding I want to use an unsupervised learning model, as the data will be unlabeled. I simply want to find and compare language patterns / recurrent keywords in text.

Useful features of unsupervised learning:
- clustering: finding structures or patterns 
- K-means: data grouping into only one cluster

TensorFlow recommend dividing the dataset into three splits: 
- [train](https://developers.google.com/machine-learning/glossary#training_set)  
- [validation](https://developers.google.com/machine-learning/glossary#validation_set)   
- [test-set](https://developers.google.com/machine-learning/glossary#test-set)  

Standardization removes punctuation or HTML elements to simplify the dataset.   
Tokenization splits strings into tokens - eg splitting a sentence into individual words.  
Vectorization converts toekns into numbers, so they can be fed into a neural network.  
This is all accomplised with the `tf.keras.layers.TextVectorization` layer.

### 2022-09-19
Looking into Jupyter notebooks as from what I remember Stefan saying, it is a great way to be able to test code at certain stages - instead of trying to debug after a run. 
Will no doubt be very valuable. 

Jupyter Notes:
- Restart: This will restart the kernels i.e. clearing all the variables that were defined, clearing the modules that were imported, etc.  
- Restart and Clear Output: This will do the same as above but will also clear all the output that was displayed below the cell.  
- Restart and Run All: This is also the same as above but will also run all the cells in the top-down order.  
- Interrupt: This option will interrupt the kernel execution. It can be useful in the case where the programs continue for execution or the kernel is stuck over some computation.  

### 2022-09-20
Find pdfs in directory tree, assign them to PdfReader object and fill into array:
```python
import os, glob
from PyPDF2 import PdfReader

pdfReader = []
def read_files():
# file path to be replaced with zotero library root
    os.chdir("/Users/tillman/t-root/dev/projects/2022/pdf-correlator/")
    for file in glob.glob("./**/*.pdf"):
        pdfReader.append(PdfReader(file))
```
Extract text from each pdf, writing it to a new [pdfTitle].txt file in 
```python
def extract_to_txt():
    os.chdir("/Users/tillman/t-root/dev/projects/2022/pdf-correlator/txt-extractions/")
    for i in pdfReaders:
        with open(i.metadata.title + ".txt", 'w') as file:
            for j in range(0, len(i.pages)):
                file.write(i.getPage(j).extract_text())
    print("text extraction complete")
```

Consider replacing file paths with `os.chdir(sys.path[0]/**)` later. `sys.path[0]` is the location of the script from where python is envoked. [see wiki](https://docs.python.org/3/library/sys.html#sys.path)

Now I have pdf sourcing, text extraction and file creation in place I need to figure out exactly what kind of ML process and model to use to achieve what I want, **which is to find language patterns and keywords in current pdfs and compare them to patterns and keywords found in incoming pdfs**

Need to look at NLP (Natural Language Processing). Don't go for BOW (Bag of Words) as it discards word order, which would take a lot of meaning out of the results in comparing two texts together - if the two pdfs were simply compared on word frequency. Instead look at Word Embedding. Word2Vec [comes highly recommended by this author](https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958), along with its GloVe extension. Also [NLTK](https://www.nltk.org/howto/semantics.html).

[use scipy to compare similarity between vectors](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html) which can be integrated with word2vec or other vector libraries for ML that transform words / sentences into vector spaces. doc2vec does sentences.

Looks like doc2vec is a viable option. It's specifically designed to detect similarites between documents - maintaining sentence structure. However, this will put a bias towards sentence structure, which may not give very useful results as academic papers are of a particular format and communication style, meaning that there may be strong correlation between papers by defaulr when looking at their sentence structure.  
Therefore, word2vec might be worth looking at first. 

### 2022-09-21
In the end, doc2vec seems best as I am in reality looking to compare documents. Not only this, but doc2vec [incorperates word2vec heavily anyway](https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e).
> "In the inference stage, a new document may be presented, and all weights are fixed to calculate the document vector."

- [doc2vec medium tutorial](https://medium.com/red-buffer/doc2vec-computing-similarity-between-the-documents-47daf6c828cd)  
- [word2vec jupyter notebook example](https://github.com/minsuk-heo/python_tutorial/blob/master/data_science/nlp/word2vec_tensorflow.ipynb)

I think I just trained a model (no errors)...
```python
import os, glob, re, io, random, gensim
import numpy as np

from PyPDF2 import PdfReader
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from nltk.tokenize import word_tokenize

pdfReaders = []
pdfFiles = []
taggedData = []
docLabels = []
wordSets = []

rootDir = "/Users/tillman/t-root/dev/projects/2022/pdf-correlator/gitignored"
txtExtractDir = "/Users/tillman/t-root/dev/projects/2022/pdf-correlator/gitignored/txt-extractions/"

# read files and extract to .txt files
def read_files():
    os.chdir(rootDir)
    for file in glob.glob("**/*.pdf"):
        pdfFiles.append(file)
        pdfReaders.append(PdfReader(file))
    print("pdf files read")
        
def extract_to_txt():
    os.chdir(txtExtractDir)
    for i in pdfReaders:
        with open(i.metadata.title + ".txt", 'w', encoding="utf-8") as file:
            
            # add doc title to array for tagging
            docLabels.append(i.metadata.title)
            print("doc labels: " + str(len(docLabels)))
            
            for j in range(0, len(i.pages)):
                # create a text file for future reference / future use
                file.write(i.getPage(j).extract_text())                   
                
    print("text extraction complete")
    
    
def clean_tag_tokenize():
    
    # read from the generated .txt files, clean and append to array
    for i in range (0, len(docLabels)):
        words = open(docLabels[i] + '.txt').read()
        words = words.replace("\n", " ")
        words = words.replace("  ", " ")
        words = words.lower()
        words = ''.join([i for i in words if i.isalpha() or i.isspace() or (i in '.!?:"')])
        words = words.replace(".", " . ")
        words = words.replace("!", " ! ")
        words = words.replace("?", " ? ")
        words = words.replace(":", " : ")
        words = words.replace('"', ' " ')
        words = words.split()
        
        wordSets.append(words)
        print("word sets: " + str(len(wordSets)))
        
    # tag the cleaned words with txt document heading (derived from pdf doc title)
    for i in range (0, len(docLabels)):
        taggedData.append(TaggedDocument(words=wordSets[i], tags=docLabels[i]))
        print(taggedData[0])

read_files()
extract_to_txt()
clean_tag_tokenize()

# The Model

model = Doc2Vec(taggedData, vector_size=5, window=2, min_count=1, workers=4)
fname = get_tmpfile("pdf_corr_doc2vec_model_0.1")

model.save(fname)
model = Doc2Vec.load(fname)
```

Replaced the above `clean_tag_tokenize()` function with the function suggested in that found in a [turorial on the Gensim website](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py). 
```python
def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

read_files()
extract_to_text()

train_corpus = list(read_corpus(txtExtractDir + docLabels[0] + ".txt")) 
for i in range (1, len(docLabels)):
    train_corpus += list(read_corpus(txtExtractDir + docLabels[i] + ".txt"))
    
print(train_corpus)
```

This gets me a much cleaner list of tagged words, that resembles the format on the Gensim code examples. However, I'm not sure if the way I've iterated through the .txt docs above (in order to fill the train_corpus as one singular object) is correct. The object does get filled and tagged successfully, but the tagging is not consecutive across the documents. ie, new-dark-age.txt gets tagged up to tag=71, then the second document is processed, but starts with the first line being tagged as 0. This means that within the training corpus, there are multiple tags 0..1..2..etc.   
I expect this will create innacurate behaviour as I expect each line in the final training corpus has to have a unique tag.

### 2022-09-22
Instead of trying to make every line (sentence / document) tag unique, I figured to prepend to the tag, the name of the .txt file. This should work just the same in terms of stopping any confusion in the vectorization, and may even give me more possibilities of data checking later down the track. 

```python
def read_corpus(fname, tokens_only=False):
        
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                for label in docLabels:
                    yield gensim.models.doc2vec.TaggedDocument(tokens, label + str(-i))
```

Indeed the improper tagging system has screwed me. I'm up to the step of assessing the model, whereby I need to iterate through the training corpus to programmatically fill other variables. The iteration is counting the ids of the tagged documents (lines in a .txt file) but as there are two txt documents in the corpus, each starting their tags at 0, the for loop is effectively counting one tag for two sentences and thereforeit is registering a missmatch between the range in which it should loop, and the content which the object has within it...    

I will need to go back to the corpus building stage. Either I need to find the way in which multiple txt files are handled properly, or I need to create one massive txt file from all other txt files and use that as the training corpus.





Concept

Motivation, idea, vision, creative / artistic / technical concept
Implementation

How did you do it? Pipeline, execution details, etc.
Results

Documentation of your result(s), e.g. images.
Project Reflection & Discussion

What worked well, what didn't work and why? In which context does your project stand? Future Work?
Lessons Learned
