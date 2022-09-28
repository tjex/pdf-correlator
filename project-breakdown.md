# Concept
The pdf-correlator is a program that analyses a given collection of local pdf files and provides means to visualise their similarity to one another.  

## Motivation, idea, vision, creative / artistic / technical concept
During the writing of my academic paper, I became interested in the idea of confirmation bias, whereby the researcher accumulates knowledge that confirms either what they already know, or what they are wanting to prove. I considered that a way to combat this could be to have a computer program that looks at your current collection of pdfs (like your Zotero library) and then looks at a new incoming pdf. From these two data sources, the program would then assess the *similarity* between the entire current library and the new pdf/s.   
If the researcher then found that the score was too similar to their current library, they could be insighted to keep searching or search more widely (or get annoyed and close the program).

## Implementation
After some research I found the Gensim doc2vec machine learning model, which was specifically designed to analyse 'documents', rather than singular words (like the word2vec model does, and from which doc2vec stems from). A 'document' in the language of this model is all text that resides on a singular line in a .txt document.   
Therefore, to implement my idea I had to go through these major steps:
1. scan for pdfs
2. iterativley extract all text from each pdf, writing the text to a singular line in a txt file
3. train the doc2vec model on the resulting text document data
4. source the vector space data resultant from the training
5. use this data to visualise the similarity between documents and/or compare the similarity to a new pdf document

## How did you do it? Pipeline, execution details, etc.
1. all pdf files read from a user set root folder using `glob.glob`, which searches through all subdirectories as well. 
2. for text extraction I used PyPDF2 and as a fallback pdfminer (as PyPDF2, while significantly faster, doesn't have as stronger compatabilty as pdfminer).
3. for the training steps I followed the standard procedure as outlined in the [Gensim doc2vec tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)
4. to acquire the vector space data (the relationship between each pdf in multidimensional vector space), I used the `model.save_word2vec_format` method (see [this jupyter notebook tutorial](https://nbviewer.org/github/RaRe-Technologies/gensim/blob/8f7c9ff4c546f84d42c220dcf28543500747c171/docs/notebooks/Tensorboard_visualizations.ipynb#Training-the-Doc2Vec-Model)
5. the process from step 4 provide two .tsv files (1x vectors, 1x metadata) which is used with [tensorboard](https://projector.tensorflow.org/) to visualise the data in a three dimensional representation of similarity. The more similar a document is to another, the closer it will be positionally to it's similar partner/s. 

## Results

## Documentation of your result(s), e.g. images.
Refer to [project log](./pdf-correlator-project-log.md).

## Project Reflection & Discussion

## What worked well, what didn't work and why? In which context does your project stand? Future Work?
## Lessons Learned
- going to the source of the library (homepage, repo, etc) before looking at user tutorials, while more confonting and difficult, is more efficient in the long run (and you learn more)
- I don't like formating code blocks without curly braces! 
- look at multiple sollutions online before trying one
- putting an incremented variable (ie a counter) inside [], and using it as variable in a function, makes it an iterable object ("see generate a training corpus" section of the code)
- people give no attention to the metadata titles in their pdfs...


