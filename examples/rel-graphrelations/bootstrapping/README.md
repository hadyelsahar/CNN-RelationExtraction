# Loading the CoreNLP Server and get it running 

- make sure you have java-8 installed
- download and unzip `stanford-corenlp-2015-12-09`
 
 ```
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip
unzip stanford-corenlp-full-2015-12-09.zip
 ```

- navigate to the folder and run the corenlp server 
- for more infor check (CoreNLP Server Webpage)[http://stanfordnlp.github.io/CoreNLP/corenlp-server.html]


```
# Set up your classpath. For example, to add all jars in the current directory tree:
export CLASSPATH="`find . -name '*.jar'`"

# Run the server
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer [port?]
```




