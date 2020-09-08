#-*- coding: utf-8 -*-
#!python3
# https://stanfordnlp.github.io/CoreNLP/download.html
# https://stanfordnlp.github.io/CoreNLP/other-languages.html#python
# https://stanfordnlp.github.io/stanza/client_usage.html

import os
import langdetect
from stanza.server import CoreNLPClient


# Setting the CoreNLP root folder as environment variable
# use .bash_profile
# export CLASSPATH=$CLASSPATH:/usr/local/StanfordCoreNLP/stanford-corenlp-4.1.0/*:
# CORENLP_HOME="/usr/local/StanfordCoreNLP/stanford-corenlp-4.1.0"
# for file in `find $CORENLP_HOME/ -name "*.jar"`;
# do export CLASSPATH="$CLASSPATH:`realpath $file`"; done

#####
# corenlp_home = os.environ['CORENLP_HOME']

########### For Chinese
# properties from StanfordCoreNLP-chinese.properties
# When interacting with the server, lists of strings are handled in parentheses, split by spaces and not commas.
def get_StanfordCoreNLP_chinese_properties():
    StanfordCoreNLP_chinese_properties = {'annotators':('tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'coref'),'tokenize.language':'zh','segment.model':'edu/stanford/nlp/models/segmenter/chinese/ctb.gz','segment.sighanCorporaDict':'edu/stanford/nlp/models/segmenter/chinese','segment.serDictionary':'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz','segment.sighanPostProcessing':True,'ssplit.boundaryTokenRegex':'[.。]|[!?！？]+','pos.model':'edu/stanford/nlp/models/pos-tagger/chinese-distsim.tagger','ner.language':'chinese','ner.model':'edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz','ner.applyNumericClassifiers':True,'ner.useSUTime':False,'ner.fine.regexner.mapping':'edu/stanford/nlp/models/kbp/chinese/gazetteers/cn_regexner_mapping.tab','ner.fine.regexner.noDefaultOverwriteLabels':'CITY,COUNTRY,STATE_OR_PROVINCE','parse.model':'edu/stanford/nlp/models/srparser/chineseSR.ser.gz','depparse.model   ':'edu/stanford/nlp/models/parser/nndep/UD_Chinese.gz','depparse.language':'chinese','coref.sieves':'ChineseHeadMatch, ExactStringMatch, PreciseConstructs, StrictHeadMatch1, StrictHeadMatch2, StrictHeadMatch3, StrictHeadMatch4, PronounMatch','coref.input.type':'raw','coref.postprocessing':True,'coref.calculateFeatureImportance':False,'coref.useConstituencyTree':True,'coref.useSemantics':False,'coref.algorithm':'hybrid','coref.path.word2vec':'','coref.language':'zh','coref.defaultPronounAgreement':True,'coref.zh.dict':'edu/stanford/nlp/models/dcoref/zh-attributes.txt.gz','coref.print.md.log':False,'coref.md.type':'RULE','coref.md.liberalChineseMD':False,'kbp.semgrex':'edu/stanford/nlp/models/kbp/chinese/semgrex','kbp.tokensregex':'edu/stanford/nlp/models/kbp/chinese/tokensregex','kbp.language':'zh','kbp.model':None,'entitylink.wikidict':'edu/stanford/nlp/models/kbp/chinese/wikidict_chinese.tsv.gz'}
    return StanfordCoreNLP_chinese_properties
# stanford-corenlp-4.1.0-models-chinese.jar

########### For English
# stanford-corenlp-4.1.0-models-english.jar
# properties are the default

# ### Annotators explained
# default annotators is all: annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref']
# tokenize: splits each word
# ssplit: splits the structure by sentence in a list of sentences.
# pos: Part of Speech tagging
# lemma: lemmatizes the words to a basic conjugation/ dictionary form
# ner: Named Entity Recognizer
# parse: ???
# depparse: Dependency Parsing
# coref: ???

def English_CoreNLPClient(text=None, annotators=None):
    if text==None:
        text = 'This is a test sentence for the server to handle. I wonder what it will do.'
    if annotators==None:
        annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref']
    with CoreNLPClient(annotators=annotators, timeout=15000) as client:
        ann = client.annotate(text)
    # sent_list = [token.word for token in ann.sentence[0].token]
    # ['This', 'is', 'a', 'test', 'sentence', 'for', 'the', 'server', 'to', 'handle','.']
    # sent_list = [token.word for token in ann.sentence[1].token]
    # ['I', 'wonder', 'what', 'it', 'will', 'do','.']
    return ann

def Chinese_CoreNLPClient(text=None, annotators=None):
    StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()
    if text==None:
        text = ("国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。")
    ####
    if annotators==None:
        annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref']
    with CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
        ann = client.annotate(text)
    # sent_list = [token.word for token in ann.sentence[0].token]
    # ['国务院', '日前', '发出', '紧急', '通知', '，', '要求', '各地', '切实', '落实', '保证', '市场', '供应', '的', '各', '项', '政策', '，', '维护', '副食品', '价格', '稳定', '。']
    return ann


##########################
##### Segmentation #######
##########################

def Segment_Chinese_only(text, sent_split=True, tolist=True):
    # Grabs a Chinese string and returns as list of words nested in a list of sentences
    # sent_split=True if we want to split the text into sentences, and then parse each sentence individually.
    # tolist=True if we want to receive a list of words, False if we want a sentence split by spaces
    StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()
    words=[]
    if text!='':
        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "undetermined"
        if (lang == "zh-cn"): #If text is Chinese, segment it, else leave it
            #########
            if sent_split:
                annotators = ['tokenize', 'ssplit']
                with CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
                    ann = client.annotate(text)
                words = [[token.word for token in sent.token] for sent in ann.sentence]
                segmented_list = [' '.join(wordlist) for wordlist in words]
                segmented = '\n'.join(segmented_list)
            else:
                annotators = ['tokenize']
                with CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
                    ann = client.annotate(text)
                words = [token.word for token in ann.sentencelessToken]
                segmented = ' '.join(words)
        else:
            segmented = text
            words = segmented.split()
    else:
        segmented = text
    if tolist:
        return words #list
    else:
        return segmented #string

def Segment(text, sent_split=True, tolist=True):
    StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()
    words=[]
    if text!='':
        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "undetermined"
        if sent_split:
            annotators = ['tokenize', 'ssplit', 'pos']
        else:
            annotators = ['tokenize','pos']
        ##########
        if (lang == "zh-cn") or (lang == "en"):
            if (lang == "zh-cn"):
                with CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
                    ann = client.annotate(text)
            elif (lang == "en"):
                with CoreNLPClient(annotators=annotators, timeout=15000) as client:
                    ann = client.annotate(text)
            #########
            if sent_split:
                words = [[token.word for token in sent.token] for sent in ann.sentence]
                segmented_list = [' '.join(wordlist) for wordlist in words]
                segmented = '\n'.join(segmented_list)
            else:
                words = [token.word for token in ann.sentencelessToken]
                segmented = ' '.join(words)
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

def POSTag_Chinese_ony(text, sent_split=True, tolist=True):
    StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()
    words=[]
    if text!='':
        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "undetermined"
        if (lang == "zh-cn"): #If text is chinese segment, else leave it
            #########
            if sent_split:
                annotators = ['tokenize', 'ssplit', 'pos']
                with CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
                    ann = client.annotate(text)
                words = [[(token.word,token.pos) for token in sent.token] for sent in ann.sentence]
                segmented_list = [' '.join(['#'.join(posted) for posted in wordlist]) for wordlist in words]
                segmented = '\n'.join(segmented_list)
            else:
                annotators = ['tokenize','pos']
                with CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
                    ann = client.annotate(text)
                words = [(token.word, token.pos) for token in ann.sentencelessToken]
                segmented = ' '.join(['#'.join(posted) for posted in words])
        else:
            segmented = text
            words = segmented.split()
    else:
        segmented = text
    if tolist:
        return words #list
    else:
        return segmented #string

def POSTag(text, sent_split=True, tolist=True):
    StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()
    words=[]
    if text!='':
        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "undetermined"
        if sent_split:
            annotators = ['tokenize', 'ssplit', 'pos']
        else:
            annotators = ['tokenize','pos']
        ##########
        if (lang == "zh-cn") or (lang == "en"):
            if (lang == "zh-cn"):
                with CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
                    ann = client.annotate(text)
            elif (lang == "en"):
                with CoreNLPClient(annotators=annotators, timeout=15000) as client:
                    ann = client.annotate(text)
            #########
            if sent_split:
                words = [[(token.word,token.pos) for token in sent.token] for sent in ann.sentence]
                segmented_list = [' '.join(['#'.join(posted) for posted in wordlist]) for wordlist in words]
                segmented = '\n'.join(segmented_list)
            else:
                words = [(token.word, token.pos) for token in ann.sentencelessToken]
                segmented = ' '.join(['#'.join(posted) for posted in words])
        else:
            segmented = text
            words = segmented.split()
    else:
        segmented = text
    if tolist:
        return words #list
    else:
        return segmented #string


def Parse(text, lang='zh-cn', annotators=None):
    StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()
    if annotators==None:
        annotators = ['tokenize', 'ssplit', 'lemma', 'pos', 'ner', 'parse', 'depparse', 'regnexer','coref']
        # annotators = ['tokenize', 'ssplit', 'lemma', 'pos', 'parse']
    if lang=='zh-cn':
        with CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
            ann = client.annotate(text)
    elif lang=='en':
        with CoreNLPClient(annotators=annotators, timeout=15000) as client:
            ann = client.annotate(text)
    return ann


if __name__ == '__main__':
    pass
