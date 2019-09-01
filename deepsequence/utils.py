"""
Utility functions
"""
import logging
import configparser
import numpy as np
from itertools import chain


def load_glove(file, dim):
    """Loads GloVe vectors in numpy array.
    Args:
        file (str): a path to a glove file.
    Return:
        dict: a dict of numpy arrays.
    """
    model = {}
    with open(file) as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            if len(vector) != dim:
                raise ValueError('glove embedding dimension: {} unmatch with params: {}'.format(len(vector), dim))
            model[word] = vector

    return model

def generate_tags(tagging_format, ner_tags):
    """
    generate word level ner tags, E.G.(B-PPL, I-PPL, L-PPL, O,....)
    :param tagging_format: bio, bioes, or biolu
    :param ner_tags: entity level ner tags E.G.(PPL, COM, GOV, GPE, ACA)
    :return:
    """

    if tagging_format not in ['bio', 'bioes', 'biolu']:
        raise ValueError('tagging_format must be either bio, bioes, or biolu')

    tags = []
    tagging_signs = None

    if tagging_format == 'bio':
        tagging_signs = ['B', 'I']

    if tagging_format == 'bioes':
        tagging_signs = ['B', 'I', 'E', 'S']

    if tagging_format == 'biolu':
        tagging_signs = ['B', 'I', 'L', 'U']

    for ner_tag in ner_tags:
        for tagging_sign in tagging_signs:
            tag = tagging_sign + '-' + ner_tag
            tags.append(tag)

    tags.append('O')

    return tags

def flatten(list_of_list):
    return list(chain.from_iterable(list_of_list))

def get_lengths(y_true_onehot_seq):
    lengths = []
    y_true_seq = np.argmax(y_true_onehot_seq, -1)
    for y in y_true_seq:
        try:
            pad_i = list(y).index(0)
            length = pad_i
        except ValueError:
            length = len(y)
        lengths.append(length)

    return lengths

def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
