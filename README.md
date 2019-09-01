deepsequence
===============================

version number: 1.0.0
author: Decheng Yang

Overview
--------

deep sequence labeling toolkit
    
Add new features:
1. support bilstm+crf
2. support bert 
3. support character level feature: char cnn or char lstm
4. support pos tag feature
5. support user defined domain dictionary (e.g. company)
6. support "BIOLU" format for sequence evaluation
7. easy to customize model by config file, see examples/ner_news_train.py 

Installation / Usage
--------------------

To install use pip:

    $ pip install deepsequence


Or clone the repo:

    $ git clone http://git.xxxx.com/deepsequence.git
    $ python setup.py install
    
Contributing
------------

TBD

Example
-------
train ner model for news:

see detail in /examples/ner_news_train.py:

use bilstm:
`python ner_news_train.py --config bilstm_parameters.json`

use bert:
`python ner_news_train.py --config bert_parameters.json`

TBD
