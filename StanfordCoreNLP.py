#-*- coding: utf-8 -*-
#!python3

import os
import langdetect
from stanza.server import CoreNLPClient

'''
For reference 
https://stanfordnlp.github.io/CoreNLP/download.html
https://stanfordnlp.github.io/CoreNLP/other-languages.html#python
https://stanfordnlp.github.io/stanza/client_usage.html

Setting the CoreNLP root folder as environment variable
use .bash_profile

export CLASSPATH=$CLASSPATH:/usr/local/StanfordCoreNLP/stanford-corenlp-4.1.0/*:
CORENLP_HOME="/usr/local/StanfordCoreNLP/stanford-corenlp-4.1.0"
for file in `find $CORENLP_HOME/ -name "*.jar"`;
do export CLASSPATH="$CLASSPATH:`realpath $file`"; done

#### if you need to use the CORENLP_HOME path for something:
corenlp_home = os.environ['CORENLP_HOME']

##############################
#### Annotators explained ####
##############################

# default annotators is all: annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref']
# tokenize: splits each word
# ssplit: splits the structure by sentence in a list of sentences.
# pos: Part of Speech tagging
# lemma: Lemmatizes the words to a basic conjugation/ dictionary form
# ner: Named Entity Recognizer
# parse: Parsing
# depparse: Dependency Parsing
# coref: Coreference Resolution


########### For English
# stanford-corenlp-4.1.0-models-english.jar
# properties are the default

# Examples of use:
def example_English():
    text = 'This is a test sentence for the server to handle. I wonder what it will do.'
    with CoreNLPClient(
                annotators=None,
                    #['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
                properties=None,
                            #{
                        #'tokenize_pretokenized': True, # Assume the text is tokenized by white space and sentence split by newline. Do not run a model.
                        #'tokenize_no_ssplit':True # Assume the sentences are split by two continuous newlines (\n\n). Only run tokenization and disable sentence segmentation.
                            #}, #You can add more or just use None
                timeout=150000
                    ) as client:
        ann = client.annotate(text)
    # ann is a Document class, broken down in Sentence objects, which each have Token objects inside.
    # For example, to take the list of tokenized words out:
    len(ann.sentence)
    # 2
    sent_list = [token.word for token in ann.sentence[0].token]
    # ['This', 'is', 'a', 'test', 'sentence', 'for', 'the', 'server', 'to', 'handle','.']
    sent_list = [token.word for token in ann.sentence[1].token]
    # ['I', 'wonder', 'what', 'it', 'will', 'do','.']

########### For Chinese
# properties from StanfordCoreNLP-chinese.properties

# Examples of use:
def example_Chinese():
    text = "国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"
    # Taken from stanford-corenlp-4.1.0-models-chinese.jar
    properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
    with CoreNLPClient(
                annotators=None,
                properties=properties,  # properties from StanfordCoreNLP-chinese.properties
                timeout=15000
                    ) as client:
        ann = client.annotate(text)
    sent_list = [token.word for token in ann.sentence[0].token]
    # ['国务院', '日前', '发出', '紧急', '通知', '，', '要求', '各地', '切实', '落实', '保证', '市场', '供应', '的', '各', '项', '政策', '，', '维护', '副食品', '价格', '稳定', '。']
'''

def get_StanfordCoreNLP_chinese_properties(properties=None):
    '''
    Exports properties taken from stanford-corenlp-4.1.0-models-chinese.jar to be able to run the Chinese models with the python client.
    
    :param (dict) properties: additional request properties (written on top of Chinese ones exported here)

    :return: Properties enabling Chinese language parsing, in addition to any in parameters.
    '''
    StanfordCoreNLP_chinese_properties = {'annotators':('tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'coref'),'tokenize.language':'zh','segment.model':'edu/stanford/nlp/models/segmenter/chinese/ctb.gz','segment.sighanCorporaDict':'edu/stanford/nlp/models/segmenter/chinese','segment.serDictionary':'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz','segment.sighanPostProcessing':True,'ssplit.boundaryTokenRegex':'[.。]|[!?！？]+','pos.model':'edu/stanford/nlp/models/pos-tagger/chinese-distsim.tagger','ner.language':'chinese','ner.model':'edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz','ner.applyNumericClassifiers':True,'ner.useSUTime':False,'ner.fine.regexner.mapping':'edu/stanford/nlp/models/kbp/chinese/gazetteers/cn_regexner_mapping.tab','ner.fine.regexner.noDefaultOverwriteLabels':'CITY,COUNTRY,STATE_OR_PROVINCE','parse.model':'edu/stanford/nlp/models/srparser/chineseSR.ser.gz','depparse.model   ':'edu/stanford/nlp/models/parser/nndep/UD_Chinese.gz','depparse.language':'chinese','coref.sieves':'ChineseHeadMatch, ExactStringMatch, PreciseConstructs, StrictHeadMatch1, StrictHeadMatch2, StrictHeadMatch3, StrictHeadMatch4, PronounMatch','coref.input.type':'raw','coref.postprocessing':True,'coref.calculateFeatureImportance':False,'coref.useConstituencyTree':True,'coref.useSemantics':False,'coref.algorithm':'hybrid','coref.path.word2vec':'','coref.language':'zh','coref.defaultPronounAgreement':True,'coref.zh.dict':'edu/stanford/nlp/models/dcoref/zh-attributes.txt.gz','coref.print.md.log':False,'coref.md.type':'RULE','coref.md.liberalChineseMD':False,'kbp.semgrex':'edu/stanford/nlp/models/kbp/chinese/semgrex','kbp.tokensregex':'edu/stanford/nlp/models/kbp/chinese/tokensregex','kbp.language':'zh','kbp.model':None,'entitylink.wikidict':'edu/stanford/nlp/models/kbp/chinese/wikidict_chinese.tsv.gz'}
    if properties:
        StanfordCoreNLP_chinese_properties.update(properties)
    return StanfordCoreNLP_chinese_properties


#### For convenience:
def Chinese_CoreNLPClient(text, annotators=None, properties=None, timeout=15000):
    properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
    with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout) as client:
        ann = client.annotate(text)
    return ann


#### For convenience:
def English_CoreNLPClient(text, annotators=None, properties=None, timeout=15000):
    with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout) as client:
        ann = client.annotate(text)
    return ann

############################################################################################
############################################################################################
############################################################################################
##### Methods to simplify using the CoreNLP client for specific purposes in my project #####
############################################################################################
############################################################################################
############################################################################################

def flatten(container):
    '''Make a list or tuple like [1,[2,3],[4,[5]]] into [1,2,3,4,5]'''
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

##########################
##### Segmentation #######
##########################

def Segment(text, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False):
    '''
    Processes a Chinese or English string and returns list of words nested in lists of sentences, or text split by spaces and newlines depending on parameters.
    
    :param (str | unicode) text: raw text for the CoreNLPServer to parse
    :param (bool) sent_split: Set True to split text into sentences. Set False to keep the text as one sentence.
    :param (bool) tolist: set to True (default) for a list of words nested in a list of sentences. Set False for a sentences split by newlines and words split by spaces.
    :param (dict) properties: additional request properties (written on top of Chinese ones exported here)
    :param (int) timeout: CoreNLP server time before raising exception.
    :param (bool) chinese_only: set to True to ignore English and other languages. Set to False to process English and Chinese. 
                                Ignoring English can save overhead, when faster tools are available.

    :return: segmented text in nested list or string

    Example:

    en_text = 'This is a test sentence for the server to handle. I wonder what it will do.'
    Segment(en_text, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False)
    >>>[['This', 'is', 'a', 'test', 'sentence', 'for', 'the', 'server', 'to', 'handle', '.'], ['I', 'wonder', 'what', 'it', 'will', 'do', '.']]

    zh_text = "国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"
    Segment(zh_text, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False)
    >>>[['国务院', '日前', '发出', '紧急', '通知', '，', '要求', '各', '地', '切实', '落实', '保证', '市场', '供应', '的', '各', '项', '政策', '，', '维护', '副食品', '价格', '稳定', '。']]
    
    Segment(zh_text, sent_split=True, tolist=False, properties=None, timeout=15000, chinese_only=False)
    >>>'国务院 日前 发出 紧急 通知 ， 要求 各 地 切实 落实 保证 市场 供应 的 各 项 政策 ， 维护 副食品 价格 稳定 。'
    
    '''
    annotators = ['tokenize', 'ssplit']
    words=[]
    if text!='':
        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "undetermined"
        if not sent_split:
            if not properties:
                properties={'tokenize_no_ssplit':True}
            else:
                properties.update({'tokenize_no_ssplit':True})
        ##########
        if chinese_only:
            segment_ok = (lang == "zh-cn")
        else:
            segment_ok = (lang == "zh-cn") or (lang == "en")
        if segment_ok:
            if (lang == "zh-cn"):
                properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
            with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout) as client:
                ann = client.annotate(text)
            words = [[token.word for token in sent.token] for sent in ann.sentence]
            segmented_list = [' '.join(wordlist) for wordlist in words]
            if sent_split:
                segmented = '\n'.join(segmented_list)
            else:
                words = flatten(words)
                segmented = ' '.join(segmented_list)
        else:
            segmented = text
            words = segmented.split()
    else:
        segmented = text
    if tolist:
        return words #list
    else:
        return segmented #string

#########################
##### POS Tagging #######
#########################

def POS_Tag(text, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False):
    '''
    Processes a Chinese or English string and returns list of words paired in tuples with their tags, nested in lists of sentences;
    or text split by spaces and newlines depending on parameters, tagged delimited by #.
    
    :param (str | unicode) text: raw text for the CoreNLPServer to parse
    :param (bool) sent_split: Set True to split text into sentences. Set False to keep the text as one sentence.
    :param (bool) tolist: set to True (default) for a list of words nested in a list of sentences. Set False for a sentences split by newlines and words split by spaces.
    :param (dict) properties: additional request properties (written on top of Chinese ones exported here)
    :param (int) timeout: CoreNLP server time before raising exception.
    :param (bool) chinese_only: set to True to ignore English and other languages. Set to False to process English and Chinese.

    POS Tags explanation

    The Chinese tags used by Stanford NLP are the same as Penn Treebank POS Tags
    
    1.  CC    Coordinating conjunction
    2.  CD    Cardinal number
    3.  DT    Determiner
    4.  EX    Existential there
    5.  FW    Foreign word
    6.  IN    Preposition or subordinating conjunction
    7.  JJ    Adjective
    8.  JJR   Adjective, comparative
    9.  JJS   Adjective, superlative
    10. LS    List item marker
    11. MD    Modal
    12. NN    Noun, singular or mass
    13. NNS   Noun, plural
    14. NNP   Proper noun, singular
    15. NNPS  Proper noun, plural
    16. PDT   Predeterminer
    17. POS   Possessive ending
    18. PRP   Personal pronoun
    19. PRP$  Possessive pronoun
    20. RB    Adverb
    21. RBR   Adverb, comparative
    22. RBS   Adverb, superlative
    23. RP    Particle
    24. SYM   Symbol
    25. TO    to
    26. UH    Interjection
    27. VB    Verb, base form
    28. VBD   Verb, past tense
    29. VBG   Verb, gerund or present participle
    30. VBN   Verb, past participle
    31. VBP   Verb, non-3rd person singular present
    32. VBZ   Verb, 3rd person singular present
    33. WDT   Wh-determiner
    34. WP    Wh-pronoun
    35. WP$   Possessive wh-pronoun
    36. WRB   Wh-adverb

    :return: segmented pairs of (word, tag) nested in sentences, or string tagged by #.

    Example:

    en_text = 'This is a test sentence for the server to handle. I wonder what it will do.'
    POS_Tag(en_text, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False)
    >>>[[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN'), ('sentence', 'NN'), ('for', 'IN'), ('the', 'DT'), ('server', 'NN'), ('to', 'TO'), ('handle', 'VB'), ('.', '.')], [('I', 'PRP'), ('wonder', 'VBP'), ('what', 'WP'), ('it', 'PRP'), ('will', 'MD'), ('do', 'VB'), ('.', '.')]]

    zh_text = "国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。"
    POS_Tag(zh_text, sent_split=True, tolist=True, properties=None, timeout=15000, chinese_only=False)
    >>>[[('国务院', 'NN'), ('日前', 'NT'), ('发出', 'VV'), ('紧急', 'JJ'), ('通知', 'NN'), ('，', 'PU'), ('要求', 'VV'), ('各', 'DT'), ('地', 'NN'), ('切实', 'AD'), ('落实', 'VV'), ('保证', 'VV'), ('市场', 'NN'), ('供应', 'NN'), ('的', 'DEG'), ('各', 'DT'), ('项', 'M'), ('政策', 'NN'), ('，', 'PU'), ('维护', 'VV'), ('副食品', 'NN'), ('价格', 'NN'), ('稳定', 'NN'), ('。', 'PU')]]
    
    POS_Tag(zh_text, sent_split=True, tolist=False, properties=None, timeout=15000, chinese_only=False)
    >>>'国务院#NN 日前#NT 发出#VV 紧急#JJ 通知#NN ，#PU 要求#VV 各#DT 地#NN 切实#AD 落实#VV 保证#VV 市场#NN 供应#NN 的#DEG 各#DT 项#M 政策#NN ，#PU 维护#VV 副食品#NN 价格#NN 稳定#NN 。#PU'

    '''
    annotators = ['tokenize', 'ssplit', 'pos']
    words=[]
    if text!='':
        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "undetermined"
        if not sent_split:
            if not properties:
                properties={'tokenize_no_ssplit':True}
            else:
                properties.update({'tokenize_no_ssplit':True})
        ##########
        if chinese_only:
            segment_ok = (lang == "zh-cn")
        else:
            segment_ok = (lang == "zh-cn") or (lang == "en")
        if segment_ok:
            if (lang == "zh-cn"):
                properties = get_StanfordCoreNLP_chinese_properties(properties=properties)
            with CoreNLPClient(annotators=annotators, properties=properties, timeout=timeout) as client:
                ann = client.annotate(text)
            words = [[(token.word,token.pos) for token in sent.token] for sent in ann.sentence]
            segmented_list = [' '.join(['#'.join(posted) for posted in wordlist]) for wordlist in words]
            if sent_split:
                segmented = '\n'.join(segmented_list)
            else:
                words = flatten(words)
                segmented = ' '.join(segmented_list)
        else:
            segmented = text
            words = segmented.split()
    else:
        segmented = text
    if tolist:
        return words #list
    else:
        return segmented #string

if __name__ == '__main__':
    pass
