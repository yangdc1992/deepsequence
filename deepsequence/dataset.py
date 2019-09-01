import os
import random
import numpy as np
import math
import json
import logging
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization import FullTokenizer


class Vocab(object):
    """
    vocabulary class
    """

    def __init__(self, vocab_pad, vocab_unk):
        """
        initial
        """
        self._padding = vocab_pad
        self._unknown = vocab_unk
        self._token2id = {self._padding: 0, self._unknown: 1}
        self._id2token = {0: self._padding, 1: self._unknown}
        
    def __len__(self):
        return len(self._token2id)

    def load(self, file_path):
        """
        build vocab from file
        :param file_name:
        :return:
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                token = line.strip()
                if token != self._padding and token != self._unknown:
                    id = len(self._token2id)
                    self._token2id[token] = id
                    self._id2token[id] = token

    @property
    def vocab(self):
        return self._token2id

    @property
    def reverse_vocab(self):
        return self._id2token

    def encode(self, token, lower=False, allow_oov=True):
        if lower:
            token = token.lower()

        if token in self._token2id:
            return self._token2id[token]
        elif allow_oov:
            return self._token2id[self._unknown]
        else:
            raise ValueError('wrong input at: {}'.format(token))

    def decode(self, idx):
        return self._id2token.get(idx, self._unknown)


class InputProcessor(object):
    def transform(self, data, mode='predict'):
        """return model input ids
        Args:
            data: data dict: e.g. {'word': [[]], 'pos': [[]], 'label': [[]]}
        returns:
            id matrixs
        """
        raise NotImplementedError()

    def inverse_transform(self, y_softmax, out_prob=False):
        """Return label strings.
        Args:
            y: label id matrix.
            out_prob: is output probability
        Returns:
            list: list of list of strings.
        """
        y_ids = np.argmax(y_softmax, -1)
        if out_prob is True:
            y_tags, y_probs = [], []
            for id_sent, sm_prob_sent in zip(y_ids, y_softmax):
                tag_sent, prob_sent = [], []
                for id, sm_prob in zip(id_sent, sm_prob_sent):
                    if id != 0:
                        tag = self.label_vocab.decode(id)
                        prob = sm_prob[id]
                        tag_sent.append(tag)
                        prob_sent.append(prob)

                y_tags.append(tag_sent)
                y_probs.append(prob_sent)

            return y_tags, y_probs
                    
        else:
            y_tags = [[self.label_vocab.decode(id) for id in id_sent if id !=0] for id_sent in y_ids]

            return y_tags

class BiLstmInputProcessor(InputProcessor):
    """
    transform inputs to idxs
    """
    def __init__(self, params):

        self.name = "bilstm_input_processor"

        self._use_char = params.use_char
        self._use_pos = params.use_pos
        self._use_dict = params.use_dict
        self._max_sent_len = params.max_sent_len
        self._max_word_len = params.max_word_len
        
        vocab_dir = params.data_dir + '/vocab'
        word_vocab_path = os.path.join(vocab_dir, 'word_vocab.txt')
        logging.info("load word vocab from :{}".format(word_vocab_path))
        self.word_vocab = Vocab(params.vocab_pad, params.vocab_unk)
        self.word_vocab.load(word_vocab_path)
        params.word_vocab_size = len(self.word_vocab)

        if self._use_char:
            char_vocab_path = os.path.join(vocab_dir, 'char_vocab.txt')
            logging.info("load char vocab from : {}".format(char_vocab_path))
            self.char_vocab = Vocab(params.vocab_pad, params.vocab_unk)
            self.char_vocab.load(char_vocab_path)
            params.char_vocab_size = len(self.char_vocab)
            
        if self._use_pos:
            pos_vocab_path = os.path.join(vocab_dir, 'pos_vocab.txt')
            logging.info('load pos vocab from: {}'.format(pos_vocab_path))
            self.pos_vocab = Vocab(params.vocab_pad, params.vocab_unk)
            self.pos_vocab.load(pos_vocab_path)
            params.pos_vocab_size = len(self.pos_vocab)

        if self._use_dict:
            dict_vocab_path = os.path.join(vocab_dir, 'dict_vocab.txt')
            logging.info('load user dict vocab from : {}'.format(dict_vocab_path))
            self.dict_vocab = Vocab(params.vocab_pad, params.vocab_unk)
            self.dict_vocab.load(dict_vocab_path)
            params.dict_vocab_size = len(self.dict_vocab)

            dict_path = params.data_dir + '/user_dict.json'
            logging.info('load user dict form : {}'.format(dict_path))
            with open(dict_path, 'r') as file:
                self.user_dict = json.load(file)

        label_vocab_path = os.path.join(vocab_dir, 'label_vocab.txt')
        logging.info('load label vocab from :{}'.format(label_vocab_path))
        self.label_vocab = Vocab(params.vocab_pad, params.vocab_unk)
        self.label_vocab.load(label_vocab_path)
        params.num_labels = len(self.label_vocab)
        self._num_labels = params.num_labels

    def transform(self, data, mode="predict"):

        word_seqs = data['word']
        word_id_seq = [[self.word_vocab.encode(word, lower=True) for word in sent] for sent in word_seqs]
        max_sent_len = max([len(sent) for sent in word_seqs])
        if max_sent_len> self._max_sent_len:
            max_sent_len = self._max_sent_len
        padded_word_id_seq = pad_sequences(word_id_seq, padding='post', maxlen=max_sent_len)

        x_inputs = [padded_word_id_seq]
        if self._use_char:
            max_word_len = max([len(word) for sent in word_seqs for word in sent[:max_sent_len]])
            if max_word_len>self._max_word_len:
                max_word_len = self._max_word_len

            padded_char_id_seq = np.zeros((data['size'], max_sent_len, max_word_len)).astype('int32')
            for i, sent in enumerate(word_seqs):
                for j, word in enumerate(sent[:max_sent_len]):
                    for k, char in enumerate(word[:max_word_len]):
                        padded_char_id_seq[i, j, k] = self.char_vocab.encode(char)
            x_inputs.append(padded_char_id_seq)

        if self._use_pos:
            pos_seqs = data['pos']
            pos_id_seq = [[self.pos_vocab.encode(pos, allow_oov=False) for pos in sent] for sent in pos_seqs]
            padded_pos_id_seq = pad_sequences(pos_id_seq, padding='post', maxlen=max_sent_len)
            x_inputs.append(padded_pos_id_seq)

        if self._use_dict:
            dict_id_seq = [[self.dict_vocab.encode(self.user_dict.get(word, 'unknown')) for word in sent] for sent in word_seqs]
            padded_dict_id_seq = pad_sequences(dict_id_seq, padding='post', maxlen=max_sent_len)
            x_inputs.append(padded_dict_id_seq)


        if mode == 'evaluate':
            label_seqs = data['label']
            label_id_seq = [[self.label_vocab.encode(label, allow_oov=False) for label in sent] for sent in label_seqs]
            y = pad_sequences(label_id_seq, padding='post', maxlen=max_sent_len)
            y = to_categorical(y, self._num_labels).astype(int)
            y = y if len(y.shape) == 3 else np.expand_dims(y, axis=0)

            return x_inputs, y

        elif mode == 'predict':
            return x_inputs

        else:
            raise ValueError("mode must be predict or evaluate")


class BertInputProcessor(InputProcessor):

    def __init__(self, params):

        self.name = 'bert_input_processor'
        self._max_sent_len = params.max_sent_len
        self._use_dict = params.use_dict

        bert_module =  hub.Module(params.bert_path)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
            self._do_lower_case = do_lower_case

        self._tokenizer = FullTokenizer(vocab_file, do_lower_case)

        label_vocab_path = params.data_dir + '/vocab/label_vocab.txt'
        logging.info('load label vocab from :{}'.format(label_vocab_path))
        self.label_vocab = Vocab(params.vocab_pad, params.vocab_unk)
        self.label_vocab.load(label_vocab_path)
        params.num_labels = len(self.label_vocab)
        self._num_labels = params.num_labels

        if params.use_dict:
            dict_vocab_path = params.data_dir + '/vocab/dict_vocab.txt'
            logging.info('load user dict vocab from : {}'.format(dict_vocab_path))
            self.dict_vocab = Vocab(params.vocab_pad, params.vocab_unk)
            self.dict_vocab.load(dict_vocab_path)
            params.dict_vocab_size = len(self.dict_vocab)

            dict_path = params.data_dir + '/user_dict.json'
            logging.info('load user dict form : {}'.format(dict_path))
            with open(dict_path, 'r') as file:
                self.user_dict = json.load(file)

    def transform(self, data, mode="predict", verbose=False):
        word_seqs = data['word']

        bert_word_seqs = []
        mapping_seqs = []

        for word_seq in word_seqs:
            bert_word_seq, mapping_seq = [], []
            for index, word in enumerate(word_seq):
                if self._do_lower_case == True:
                    word = word.lower()
                temp_words = self._tokenizer.wordpiece_tokenizer.tokenize(word)
                for i, temp_word in enumerate(temp_words):
                    bert_word_seq.append(temp_word)
                    if i==0:
                        mapping_seq.append(index)
                    else:
                        mapping_seq.append('X')
            bert_word_seq = ["[CLS]"] + bert_word_seq[:self._max_sent_len-2] + ["[SEP]"]
            mapping_seq = ["[CLS]"] + mapping_seq[:self._max_sent_len-2] + ["[SEP]"]

            bert_word_seqs.append(bert_word_seq)
            mapping_seqs.append(mapping_seq)

        data['bert_mapping'] = mapping_seqs

        bert_word_id_seqs = [self._tokenizer.convert_tokens_to_ids(sent) for sent in bert_word_seqs]
        bert_mask_id_seqs = [[1]*len(sent) for sent in bert_word_seqs]
        bert_segment_id_seqs = [[0]*len(sent) for sent in bert_word_seqs]
        
        padded_bert_word_id_seqs = pad_sequences(bert_word_id_seqs, padding='post', maxlen=self._max_sent_len)
        padded_bert_mask_id_seqs = pad_sequences(bert_mask_id_seqs, padding='post', maxlen=self._max_sent_len)
        padded_bert_segment_id_seqs = pad_sequences(bert_segment_id_seqs, padding='post', maxlen=self._max_sent_len)
        

        x_inputs = [padded_bert_word_id_seqs, padded_bert_mask_id_seqs, padded_bert_segment_id_seqs]

        if self._use_dict:
            bert_dict_tag_seqs = []
            for word_seq, mapping_seq in zip(word_seqs, mapping_seqs):
                bert_dict_tag_seq = []
                for mapping_i in mapping_seq:
                    if mapping_i in {"[CLS]", "[SEP]", "X"}:
                        dict_tag = mapping_i
                    else:
                        dict_tag = self.user_dict.get(word_seq[mapping_i], self.dict_vocab._unknown)
                    bert_dict_tag_seq.append(dict_tag)
                bert_dict_tag_seqs.append(bert_dict_tag_seq)

            bert_dict_id_seq = [[self.dict_vocab.encode(dict_tag) for dict_tag in dict_tag_sent] for dict_tag_sent in bert_dict_tag_seqs]
            padded_dict_id_seq = pad_sequences(bert_dict_id_seq, padding='post', maxlen=self._max_sent_len)
            x_inputs.append(padded_dict_id_seq)

        assert len(padded_bert_word_id_seqs[0]) == self._max_sent_len
        assert len(padded_bert_mask_id_seqs[0]) == self._max_sent_len
        assert len(padded_bert_segment_id_seqs[0]) == self._max_sent_len


        if mode == 'evaluate':
            label_seqs = data['label']
            bert_label_seqs = []
            for label_seq, mapping_seq in zip(label_seqs, mapping_seqs):
                bert_label_seq = []
                for mapping_i in mapping_seq:
                    if mapping_i in {"[CLS]", "[SEP]", "X"}:
                        bert_label_seq.append(mapping_i)
                    else:
                        bert_label_seq.append(label_seq[mapping_i])

                bert_label_seqs.append(bert_label_seq)

            bert_label_id_seqs = [[self.label_vocab.encode(label, allow_oov=False) for label in sent] for sent in bert_label_seqs]
            padded_bert_label_id_seqs = pad_sequences(bert_label_id_seqs, padding='post', maxlen=self._max_sent_len)
            y_seqs = to_categorical(padded_bert_label_id_seqs, self._num_labels).astype(int)
            y_seqs = y_seqs if len(y_seqs.shape) == 3 else np.expand_dims(y_seqs, axis=0)

            return x_inputs, y_seqs

        elif mode == 'predict':
            return x_inputs
        else:
            raise ValueError('mode must be predict or evaluate')


def load_conll(file_path, params, max_iter=None):
    """
    load conll format data
    """

    all_word_sents, all_label_sents = [], []
    word_sentence, label_sentence = [], []

    is_use_pos = params.dict.get('use_pos', False)
    if is_use_pos:
        pos_sentence = []
        all_pos_sents = []
    
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            if max_iter is not None and len(all_word_sents) == max_iter:
                break

            line = line.rstrip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if word_sentence and label_sentence:
                    all_word_sents.append(word_sentence)
                    all_label_sents.append(label_sentence)
                    word_sentence, label_sentence = [], []
                    if is_use_pos:
                        all_pos_sents.append(pos_sentence)
                        pos_sentence = []

            else:
                tup = line.split(' ')
                if not tup[0]:
                    tup = (' ', 'SPACE', tup[-1])
                word = tup[0]
                label = tup[-1]
                word_sentence.append(word)
                label_sentence.append(label)
                if is_use_pos:
                    pos = tup[params.conll_pos_index]
                    pos_sentence.append(pos)

    if word_sentence and label_sentence:
        all_word_sents.append(word_sentence)
        all_label_sents.append(label_sentence)
        if is_use_pos:
            all_pos_sents.append(pos_sentence)

    data = {'word': all_word_sents, 'label': all_label_sents, 'size': len(all_word_sents)}
    if is_use_pos:
        data['pos'] = all_pos_sents

    return data


class DataIterator(Sequence):
    """
    data iterator for training
    """

    def __init__(self, data, batch_size, input_processor, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.input_processor = input_processor
        self.indexs = np.arange(data['size'])
        self.shuffle = shuffle

    def __getitem__(self, idx):
        batch_data = {}
        for key in self.data:
            if key != 'size':
                batch_data[key] = self.data[key][idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_data['size'] = len(batch_data['word'])

        return self.input_processor.transform(batch_data, mode='evaluate')

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexs)

    def __len__(self):
        return math.ceil(self.data['size'] / self.batch_size)

