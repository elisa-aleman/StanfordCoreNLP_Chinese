# StanfordCoreNLP Chinese and English

A simplified implementation of the Python official interface Stanza for Stanford CoreNLP Java server application to parse, tokenize, part-of-speech tag Chinese and English texts.

The Stanford NLP group has released a unified language tool called CoreNLP which acts as a parser, tokenizer, part-of-speech tagger and more. They have different models for a few languages, but I use Chinese and English. These software releases are all done in Java, and while there is an official python client available, called Stanza, it is often hard to find information on how to set the software to work properly. I use langdetect and define simple functions to import easily instead of focusing on setting up the program every time. I also made a few methods to get the output I really want from the annotation results, which are very complicated and full of information that is offuscating the main objective of using the application.

This tutorial is written for Debian-based Linux systems and MacOSX. I researched Windows installing instructions and wrote them, but can't test them myself.

First: Java is necessary to run all these programs.

## Install Java Development Kit

https://www.ntu.edu.sg/home/ehchua/programming/howto/JDK_HowTo.html 

To see if previously installed

`javac -version`

Download Java SE

http://www.oracle.com/technetwork/java/javase/downloads/index.html


## Install Stanford CoreNLP

Previous standalone Stanford NLP software is being deprecated and suggest using the new and integrated CoreNLP server tool.

Download CoreNLP 4.1.0 and Unzip it somewhere:

(Debian-based Linux and MacOS X)
```
cd /usr/local/
mkdir StanfordCoreNLP
cd StanfordCoreNLP

wget http://nlp.stanford.edu/software/stanford-corenlp-latest.zip
unzip stanford-corenlp-latest.zip
cd stanford-corenlp-4.1.0
```

The root folder is then, for example:

`/usr/local/StanfordCoreNLP/stanford-corenlp-4.1.0`

Also download Chinese and English models to the root folder above:
```
cd /usr/local/StanfordCoreNLP/stanford-corenlp-4.1.0
wget http://nlp.stanford.edu/software/stanford-corenlp-4.1.0-models-chinese.jar
wget http://nlp.stanford.edu/software/stanford-corenlp-4.1.0-models-english.jar
```

You can confirm the links and versions in the follwing link:

https://stanfordnlp.github.io/CoreNLP/download.html 

Following the getting started:

Add all the .jar files to the CLASSPATH and the root folder to CORENLP_HOME

### Debian-based Linux

add the following to `/etc/profile` for system wide installation or to `~/.bash_profile` for user installation

```
CORENLP_HOME="/usr/local/StanfordCoreNLP/stanford-corenlp-4.1.0"
for file in `find $CORENLP_HOME -name "*.jar"`; do export
CLASSPATH="$CLASSPATH:`realpath $file`"; done
```

### MacOSX
In MacOSX we don’t have the ‘realpath’ module so install as a part of GNU coreutils with homebrew

`brew install coreutils`

Now we can do as in the Debian-based Linux step:

add the following to `/etc/profile` for system wide installation or to `~/.bash_profile` for user installation

```
CORENLP_HOME="/usr/local/StanfordCoreNLP/stanford-corenlp-4.1.0"
for file in `find $CORENLP_HOME -name "*.jar"`; do export
CLASSPATH="$CLASSPATH:`realpath $file`"; done
```
### Windows

I don't have a windows console available so this is untested, but it should be as follows:

Set an environment variable to the root folder called CORENLP_HOME

```
CORENLP_HOME=%HOMEDRIVE%\StanfordCoreNLP\stanford-corenlp-4.1.0
cd %CORENLP_HOME%
FOR %i IN (*.jar) DO set classpath= %classpath%;%cd%\%i
```

## Running Stanford CoreNLP Server

Run the server at the root directory, but pointing to the jar files through the CLASSPATH

Run a server using Chinese properties

### Debian-based Linux and MacOSX

```
cd $CORENLP_HOME
java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9000 -timeout 15000
```

### Windows

```
cd %CORENLP_HOME%
java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9000 -timeout 15000
```

Voilà! You now have a Stanford CoreNLP server running on your machine... as a java server.

## Python setup environment

Now for Python we use the official release of the python interface for CoreNLP at:

https://stanfordnlp.github.io/CoreNLP/other-languages.html#python

The official Stanford NLP Python package is Stanza: https://stanfordnlp.github.io/stanza/

```
pip install stanza
```
Stenza will use the $CORENLP_HOME environment variable, so as long as we set that up we should be fine.

To use it: https://stanfordnlp.github.io/stanza/client_usage.html

```
from stanza.server import CoreNLPClient

text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
        timeout=30000,
        memory='16G') as client:
    ann = client.annotate(text)
```

The CoreNLP server will be automatically started in the background upon the instantiation of the client, so normally you don’t need to worry about it.

How to parse Chinese:

```
# My own method to export properties of the Chinese model:
StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()

with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
        properties=StanfordCoreNLP_chinese_properties,
        timeout=30000,
        memory='16G') as client:
    ann = client.annotate(text)
```

The steps below are already implemented in my StanfordCoreNLP.py, but here I explain what is behind each step.

### Properties from StanfordCoreNLP-chinese.properties

I had to get the Java properties file and convert it to a python dictionary that the wrapper program can read.
The properties file is inside the chinese models .jar `$CORENLP_HOME/stanford-corenlp-4.1.0-models-chinese.jar`
So I had to unzip it and find it in `StanfordCoreNLP-chinese.properties`.
After editing the file it looks like this:

```
# properties set to chinese
# properties from StanfordCoreNLP-chinese.properties
StanfordCoreNLP_chinese_properties = {'annotators':('tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'coref'),
    'tokenize.language':'zh',
    'segment.model':'edu/stanford/nlp/models/segmenter/chinese/ctb.gz',
    'segment.sighanCorporaDict':'edu/stanford/nlp/models/segmenter/chinese',
    'segment.serDictionary':'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz',
    'segment.sighanPostProcessing':True,
    'ssplit.boundaryTokenRegex':'[.。]|[!?！？]+',
    'pos.model':'edu/stanford/nlp/models/pos-tagger/chinese-distsim.tagger',
    'ner.language':'chinese',
    'ner.model':'edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz',
    'ner.applyNumericClassifiers':True,
    'ner.useSUTime':False,
    'ner.fine.regexner.mapping':'edu/stanford/nlp/models/kbp/chinese/gazetteers/cn_regexner_mapping.tab',
    'ner.fine.regexner.noDefaultOverwriteLabels':'CITY,COUNTRY,STATE_OR_PROVINCE',
    'parse.model':'edu/stanford/nlp/models/srparser/chineseSR.ser.gz',
    'depparse.model   ':'edu/stanford/nlp/models/parser/nndep/UD_Chinese.gz',
    'depparse.language':'chinese',
    'coref.sieves':'ChineseHeadMatch, ExactStringMatch, PreciseConstructs, StrictHeadMatch1, StrictHeadMatch2, StrictHeadMatch3, StrictHeadMatch4, PronounMatch',
    'coref.input.type':'raw',
    'coref.postprocessing':True,
    'coref.calculateFeatureImportance':False,
    'coref.useConstituencyTree':True,
    'coref.useSemantics':False,
    'coref.algorithm':'hybrid',
    'coref.path.word2vec':'',
    'coref.language':'zh',
    'coref.defaultPronounAgreement':True,
    'coref.zh.dict':'edu/stanford/nlp/models/dcoref/zh-attributes.txt.gz',
    'coref.print.md.log':False,
    'coref.md.type':'RULE',
    'coref.md.liberalChineseMD':False,
    'kbp.semgrex':'edu/stanford/nlp/models/kbp/chinese/semgrex',
    'kbp.tokensregex':'edu/stanford/nlp/models/kbp/chinese/tokensregex',
    'kbp.language':'zh',
    'kbp.model':None,
    'entitylink.wikidict':'edu/stanford/nlp/models/kbp/chinese/wikidict_chinese.tsv.gz'}
```

I added this dictionary to the `StanfordCoreNLP.py` file, so it should not be necessary for you.

```
StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()
```

### Library import 

After this, place StanfordCoreNLP.py in your current directory, and import normally. For example:

```
from StanfordCoreNLP import Chinese_CoreNLPClient
```

Or put it in any folder you like (for example: ~/PersonalLibraries) and, for example, do:

```
from PersonalLibraries.StanfordCoreNLP import Chinese_CoreNLPClient
```
or 
```
from PersonalLibraries.StanfordCoreNLP import *
```

Now we can freely call the methods in this library and parse Chinese and English text from python.

```
text = "国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"
ann = Chinese_CoreNLPClient(text)
sent_list = [token.word for token in ann.sentence[0].token]
# ['国务院', '日前', '发出', '紧急', '通知', '，', '要求', '各地', '切实', '落实', '保证', '市场', '供应', '的', '各', '项', '政策', '，', '维护', '副食品', '价格', '稳定', '。']
```

Or in case you want to set up the Client by yourself:

```
from StanfordCoreNLP import get_StanfordCoreNLP_chinese_properties
StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()

with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
        properties=StanfordCoreNLP_chinese_properties,
        timeout=30000,
        memory='16G') as client:
    ann = client.annotate(text)
```
