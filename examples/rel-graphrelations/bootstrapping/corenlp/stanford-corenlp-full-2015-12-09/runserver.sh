# Set up your classpath. For example, to add all jars in the current directory tree:
export CLASSPATH="`find . -name '*.jar'`"

# Run the server
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
