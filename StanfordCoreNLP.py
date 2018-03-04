#-*- coding: utf-8 -*-
#!python3

import os
import langdetect
import corenlp

# Setting the CoreNLP root folder as environment variable
corenlp_home = os.path.join(os.path.expanduser('~'),'StanfordCoreNLP','stanford-corenlp-full-2018-02-27')
os.environ['CORENLP_HOME'] = corenlp_home
# properties from StanfordCoreNLP-chinese.properties
# When interacting with the server, lists of strings are handled in parentheses, split by spaces and not commas.
StanfordCoreNLP_chinese_properties = {'annotators':('tokenize' 'ssplit' 'pos' 'lemma' 'ner' 'parse' 'mention' 'coref'),'tokenize.language':'zh','segment.model':'edu/stanford/nlp/models/segmenter/chinese/ctb.gz','segment.sighanCorporaDict':'edu/stanford/nlp/models/segmenter/chinese','segment.serDictionary':'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz','segment.sighanPostProcessing':True,'ssplit.boundaryTokenRegex':'[.。]|[!?！？]+','pos.model':'edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger','ner.language':'chinese','ner.model':'edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz','ner.applyNumericClassifiers':True,'ner.useSUTime':False,'regexner.mapping':'edu/stanford/nlp/models/kbp/cn_regexner_mapping.tab','regexner.validpospattern':'^(NR|NN|JJ).*','regexner.ignorecase':True,'regexner.noDefaultOverwriteLabels':'CITY','parse.model':'edu/stanford/nlp/models/srparser/chineseSR.ser.gz','depparse.model':'edu/stanford/nlp/models/parser/nndep/UD_Chinese.gz','depparse.language':'chinese','coref.sieves':('ChineseHeadMatch' 'ExactStringMatch' 'PreciseConstructs' 'StrictHeadMatch1' 'StrictHeadMatch2' 'StrictHeadMatch3' 'StrictHeadMatch4' 'PronounMatch'),'coref.input.type':'raw','coref.postprocessing':True,'coref.calculateFeatureImportance':False,'coref.useConstituencyTree':True,'coref.useSemantics':False,'coref.algorithm':'hybrid','coref.path.word2vec':'','coref.language':'zh','coref.defaultPronounAgreement':True,'coref.zh.dict':'edu/stanford/nlp/models/dcoref/zh-attributes.txt.gz','coref.print.md.log':False,'coref.md.type':'RULE','coref.md.liberalChineseMD':False,'kbp.semgrex':'edu/stanford/nlp/models/kbp/chinese/semgrex','kbp.tokensregex':'edu/stanford/nlp/models/kbp/chinese/tokensregex','kbp.model':None,'entitylink.wikidict':'edu/stanford/nlp/models/kbp/wikidict_chinese.tsv.gz'}

def English_CoreNLP_test(text=None, annotators=None):
    # corenlp_home = os.path.join(os.path.expanduser('~'),'StanfordCoreNLP','stanford-corenlp-full-2018-02-27')
    # os.environ['CORENLP_HOME'] = corenlp_home
    # ####
    if text==None:
        text = 'This is a test sentence for the server to handle. I wonder what it will do.'
    ####
    # default annotators is all: ['tokenize', 'ssplit', 'lemma', 'pos', 'ner', 'depparse']
    # tokenize: splits each word
    # ssplit: splits the structure by sentence in a list of sentences.
    # lemma: lemmatizes the words to a basic conjugation/ dictionary form
    # pos: Part of Speech tagging
    # ner: Named Entity Recognizer
    # depparse: Dependency Parsing
    if annotators==None:
        annotators = ["tokenize", "ssplit"]
        # annotators = ['tokenize', 'ssplit', 'lemma', 'pos', 'ner', 'depparse']
    with corenlp.CoreNLPClient(annotators=annotators, timeout=15000) as client:
        ann = client.annotate(text)
    # sent_list = [token.word for token in ann.sentence[0].token]
    # ['This', 'is', 'a', 'test', 'sentence', 'for', 'the', 'server', 'to', 'handle','.']
    # sent_list = [token.word for token in ann.sentence[1].token]
    # ['I', 'wonder', 'what', 'it', 'will', 'do','.']
    return ann

def Chinese_CoreNLP_test(text=None, annotators=None):
    # corenlp_home = os.path.join(os.path.expanduser('~'),'StanfordCoreNLP','stanford-corenlp-full-2018-02-27')
    # os.environ['CORENLP_HOME'] = corenlp_home
    ####
    if text==None:
        text = ("国务院日前发出紧急通知，要求各地切实落实保证市场供应的各项政策，维护副食品价格稳定。")
    ####
    if annotators==None:
        annotators = ['tokenize', 'ssplit', 'pos']
        # annotators = ['tokenize', 'ssplit', 'lemma', 'pos', 'ner', 'depparse']
    with corenlp.CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
        ann = client.annotate(text)
    # sent_list = [token.word for token in ann.sentence[0].token]
    # ['国务院', '日前', '发出', '紧急', '通知', '，', '要求', '各地', '切实', '落实', '保证', '市场', '供应', '的', '各', '项', '政策', '，', '维护', '副食品', '价格', '稳定', '。']
    return ann

#Grabs a chinese string and returns as list of words nested in a list of sentences
def Segment(text, sent_split=True, tolist=True):
    words=[]
    if text!='':
        try:
            lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "undetermined"
        if (lang == "zh-cn"): #If text is chinese segment, else leave it
            #########
            if sent_split:
                annotators = ['tokenize', 'ssplit']
                with corenlp.CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
                    ann = client.annotate(text)
                words = [[token.word for token in sent.token] for sent in ann.sentence]
                segmented_list = [' '.join(wordlist) for wordlist in words]
                segmented = '\n'.join(segmented_list)
            else:
                annotators = ['tokenize']
                with corenlp.CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
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

def POSTag(text, sent_split=True, tolist=True):
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
                with corenlp.CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
                    ann = client.annotate(text)
                words = [[(token.word,token.pos) for token in sent.token] for sent in ann.sentence]
                segmented_list = [' '.join(['#'.join(posted) for posted in wordlist]) for wordlist in words]
                segmented = '\n'.join(segmented_list)
            else:
                annotators = ['tokenize','pos']
                with corenlp.CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
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

def Parse(text, annotators=None):
    if annotators==None:
        # annotators = ['tokenize', 'ssplit', 'lemma', 'pos', 'ner', 'parse', 'depparse', 'regnexer','coref']
        annotators = ['tokenize', 'ssplit', 'lemma', 'pos', 'parse']
    with corenlp.CoreNLPClient(annotators=annotators, properties=StanfordCoreNLP_chinese_properties, timeout=15000) as client:
        ann = client.annotate(text)
    return ann

if __name__ == '__main__':
    pass
