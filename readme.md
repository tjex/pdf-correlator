Here is a code walkthrough and demonstration [on Youtube](https://youtu.be/_js_7dmWIuw)

# Concept
The pdf-correlator is a program that analyses a given collection of local pdf files and provides means to visualise their similarity to one another. Additionally, it can compare the similarity score between the entire collection and a singular incoming (new) pdf. 

## Motivation, idea, vision, creative / artistic / technical concept
During the writing of my academic paper, I became interested in the idea of confirmation bias, whereby the researcher accumulates knowledge that confirms either what they already know, or what they are wanting to prove. I considered that a way to combat this could be to have a computer program that looks at your current collection of pdfs (like your Zotero library) and then looks at a new incoming pdf. From these two data sources, the program could then infer the *similarity* between the entire current library and the new pdf/s.   
If the researcher then found that the score was too similar to their current library, they could be insighted to search more widely (or get annoyed and close the program).

## Implementation
After some research I found the Gensim doc2vec machine learning model, which was specifically designed to analyse 'documents', rather than singular words (like the word2vec model does, and from which doc2vec stems from). A 'document' in the language of this model is all text that resides on a singular line in a .txt document. The model derives patterns from the text it is given and vectorizes the data, meaning that it can then have relationships inferred by comparing the distances and angles between other vectorized 'documents' in vector space.

To implement my idea I had to go through these major steps:
1. scan for pdfs
2. iterativley extract all text from each pdf, writing the text to a singular line in a txt file
3. train the doc2vec model on the resulting text document data
4. acquire the vector space data resultant from the training
5. use this data to visualise the similarity between documents and/or compare the similarity to a new pdf document

## How did you do it? Pipeline, execution details, etc.
1. all pdf files read from a user set root folder using `glob.glob` with recursivity, allowing searches through all subdirectories as well. 
2. for text extraction I used PyPDF2 and pdfminer as an error handling fallback (as PyPDF2, while significantly faster, doesn't have as stronger compatabilty as pdfminer).
3. for the training steps I followed the standard procedure as outlined in the [Gensim doc2vec tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)
4. to acquire the vector space data (the relationship between each pdf in multidimensional vector space), I used the `model.save_word2vec_format` method learned from [this jupyter notebook tutorial](https://nbviewer.org/github/RaRe-Technologies/gensim/blob/8f7c9ff4c546f84d42c220dcf28543500747c171/docs/notebooks/Tensorboard_visualizations.ipynb#Training-the-Doc2Vec-Model).
5. the process from step 4 provides two .tsv files (1x vectors, 1x metadata) which are used with [tensorboard](https://projector.tensorflow.org/) to visualise the data in a three dimensional representation of similarity. The more similar a document is to another, the closer it will be positionally to it's similar partner/s. 
6. comparing a new incoming pdf to the corpus as a whole is done with the `infer_vector` method. See the very last code block in the pdf-correlator.script

## Results
An ML model trained on your library of pdfs, a way to visualize the relationships of similarites between all pdfs that the ML model was trained on, and a process to allow the inferring of similarity between a new pdf (ie a pdf that the ML model hasn't had as part of its training) and your pdf library.

# Development / Running
dependencies (version used):
  - python (3.10.6)
  - tensorflow (2.10)
  - pypdf2 (2.10.8)
  - gensim (4.2.0)
  - pdfminer.six (20220524)
  - jupyter (for running via jupyter notebook - recommended)

If you just want to use this script as is, you just need to install the above dependancies and change the `pdfReadRootDir` to a folder from which you'd like to pull in pdfs (eg the root directory of your Zotero library).

If you are running from a terminal, be sure to invoke the script with `ipython3` instead of `python3`, as `ipython3` is required for a script call when generating the .tsv files for tensorboard.

