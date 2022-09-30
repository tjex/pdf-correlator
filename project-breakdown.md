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

## Results
An ML model trained on your library of pdfs, a way to visualize the relationships of similarites between all pdfs that the ML model was trained on, and a process to allow the inferring of similarity between a new pdf (ie a pdf that the ML model hasn't had as part of its training) and your pdf library.

## Documentation of your result(s), e.g. images.
Here is a walkthrough video showing the process and results. 
For extra documentation, please refer to [project log](./pdf-correlator-project-log.md).

## Project Reflection & Discussion

### What worked well, what didn't work and why? 
In terms of working / not working, nothing was on either extreme end of the spectrum.  
There were of course some set backs and problems:  
- Training the model on such a small set of data did not allow for useable self-similarity testing results. So, while the training went fine, and the model functions, I can't be sure to what degree of clarity / effectiveness the model is functioning. 
- The PyPDF2 package had trouble reading some pdfs (eg scanned documents), so I needed to set up an exception handler to extract with pdfminer instead.
- At the time of writing, I haven't figured out how to execute a python script from within a python script (getting permission errors). From within jupyter notebook it works.
- It was challenging to process the text data correctly for the model to be in line with my desired outcome. There was a fair amount of manually looking through generated training data text files to check if things were looking ok.
- One thing that I could never figure out was how to provide two tags to a document (ie, pdf name + line index). It is apparently possible.

### In which context does your project stand? Future Work?
This project lives in the context of data science and user apps.  
On the one hand, by it's very nature the ML model is a useful tool to understand patterns and relationships that are not immediately (or at all) apparent to us. In the case of this project, this pattern finding ability is put to practice on a user's own collection of pdf files. I see this as being a useful supplementary tool to researchers and writers alike who would like to gain an insight into the similarity relationships between local pdf files on their machine. 

The original intention was to point the pdf search directory to the root of a program like Zotero, but due to the file search procedure simply looking for any \*.pdf file within a directory and all subsequent subdirectories, there's no reason why this program could also be used to uncover similarity relationships between pdf documents adhearing to other topics where similarity relationships could be useful (eg, plagiarism, census data, historical data, etc).

If I am to continue working on this I would like to build the script into an app, allowing interaction through a GUI rather than having to hard code directory locations and run blocks of code via jupyter notebook / terminal. 


### Lessons Learned
- going to the source of the library (homepage, repo, etc) before looking at user tutorials, while more confonting and difficult, is more efficient in the long run (and you learn more)
- I don't like formating code blocks without curly braces! 
- look at multiple sollutions online (and understand the question and suggested sollution) before trying one
- putting an incremented variable (ie a counter) inside [], and using it as variable in a function, makes it an iterable object ("see generate a training corpus" section of the code)
- people give no attention to the metadata titles of their pdfs...
- text cleaning is a really difficult procedure
- vectors are proving again and again to be a super versatile and really powerfull way of working with data.


