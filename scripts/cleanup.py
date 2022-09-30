import os, glob

for file in glob.glob("./model-data/*.tsv"):
    os.remove(file)

for file in glob.glob("./model-data/*.txt"):
    os.remove(file)

for file in glob.glob("./model-data/*.w2v"):
    os.remove(file)
 
for file in glob.glob("./txt-extractions/*.txt"):
    os.remove(file)

for file in glob.glob("./tests/*.txt"):
    os.remove(file)    
