"""
Model definition.
"""
import logging

import numpy as np
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, add, TimeDistributed
from keras.layers.merge import Concatenate
from keras.models import Model, load_model, model_from_json
from deepsequence.layers import CRF, LayerNormalization, char_cnn_encode, BertLayer
from deepsequence.dataset import BiLstmInputProcessor, DataIterator, BertInputProcessor
from deepsequence.utils import load_glove, get_lengths
from deepsequence.callbacks import F1score
from deepsequence.seqeval import f1_score, classification_report
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf
from bert.optimization import AdamWeightDecayOptimizer


class SequenceModel(object):

    def __init__(self, params):
        self.input_processor = None
        self.model = None

        if params.model == "bilstm":
            self.input_processor = BiLstmInputProcessor(params)
        elif params.model == "bert":
            self.input_processor = BertInputProcessor(params)
        else:
            raise ValueError("model must be bilstm or bert")

        self._params = params
        self.build()

    def build(self, verbose=True):
        if self._params.model == 'bilstm':
            self.build_bilstm(verbose)

        if self._params.model == "bert":
            self.build_bert(verbose)

    def build_bilstm(self, verbose=True):
        """
        build model architecture from parameters
        """
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        inputs = [word_ids]

        if self._params.use_pretrain_embedding:
            if verbose: logging.info("initial word embedding with pretrained embeddings")
            if self._params.word_embedding_dim == 100:
                glove_file = self._params.data_dir + '/glove.6B.100d.txt'
            elif self._params.word_embedding_dim == 300:
                glove_file = self._params.data_dir + '/glove.42B.300d.txt'
            else:
                logging.error("we only support glove embedding with dimension 100 or 300")
                raise ValueError("unmatch word dimension, we only support glove embedding with dimension 100 or 300")
            glove_embedding_index = load_glove(glove_file, self._params.word_embedding_dim)
            word_vocab = self.input_processor.word_vocab.vocab
            glove_embeddings_matrix = np.zeros([len(word_vocab), self._params.word_embedding_dim])
            for word, i in word_vocab.items():
                vector = glove_embedding_index.get(word)
                if vector is not None:
                    glove_embeddings_matrix[i] = vector
            
            word_embeddings = Embedding(input_dim=glove_embeddings_matrix.shape[0],
                                        output_dim=glove_embeddings_matrix.shape[1],
                                        trainable=False,
                                        mask_zero=True,
                                        weights=[glove_embeddings_matrix],
                                        name='word_embedding')(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._params.word_vocab_size,
                                        output_dim=self._params.word_embedding_dim,
                                        mask_zero=True,
                                        name='word_embedding')(word_ids)

        input_embeddings = [word_embeddings]
        if self._params.use_char:
            char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
            inputs.append(char_ids)
            if self._params.char_feature == "lstm":
                char_embeddings = Embedding(input_dim=self._params.char_vocab_size,
                                            output_dim=self._params.char_embedding_dim,
                                            mask_zero=True,
                                            name='char_embedding')(char_ids)
                if verbose: logging.info("using charcter level lstm features")
                char_feas = TimeDistributed(Bidirectional(LSTM(self._params.char_lstm_size)), name="char_lstm")(char_embeddings)
            elif self._params.char_feature == "cnn":
                # cnn do not support mask
                char_embeddings = Embedding(input_dim=self._params.char_vocab_size,
                                            output_dim=self._params.char_embedding_dim,
                                            name='char_embedding')(char_ids)
                if verbose: logging.info("using charcter level cnn features")
                char_feas = char_cnn_encode(char_embeddings, self._params.n_gram_filter_sizes, self._params.n_gram_filter_nums)
            else:
                raise ValueError('char feature must be lstm or cnn')

            input_embeddings.append(char_feas)

        if self._params.use_pos:
            if verbose: logging.info("use pos tag features")
            pos_ids = Input(batch_shape=(None, None), dtype='int32', name='pos_input')
            inputs.append(pos_ids)


            pos_embeddings = Embedding(input_dim=self._params.pos_vocab_size,
                                        output_dim=self._params.pos_embedding_dim,
                                        mask_zero=True,
                                        name='pos_embedding')(pos_ids)
            input_embeddings.append(pos_embeddings)

        if self._params.use_dict:
            if verbose: logging.info("use user dict features")
            dict_ids = Input(batch_shape=(None, None), dtype='int32', name='dict_input')
            inputs.append(dict_ids)

            dict_embeddings = Embedding(input_dim=self._params.dict_vocab_size,
                                        output_dim=self._params.dict_embedding_dim,
                                        mask_zero=True,
                                        name='dict_embedding')(dict_ids)
            input_embeddings.append(dict_embeddings)

        input_embedding = Concatenate(name="input_embedding")(input_embeddings) if len(input_embeddings)>1 else input_embeddings[0]
        input_embedding_ln = LayerNormalization(name='input_layer_normalization')(input_embedding)
        #input_embedding_bn = BatchNormalization()(input_embedding_ln)
        input_embedding_drop = Dropout(self._params.dropout, name="input_embedding_dropout")(input_embedding_ln)

        z = Bidirectional(LSTM(units=self._params.main_lstm_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                           name="main_bilstm")(input_embedding_drop)
        z = Dense(self._params.fc_dim, activation='tanh', name="fc_dense")(z)

        if self._params.use_crf:
            if verbose: logging.info('use crf decode layer')
            crf = CRF(self._params.num_labels, sparse_target=False,
                        learn_mode='marginal', test_mode='marginal', name='crf_out')
            loss = crf.loss_function
            pred = crf(z)
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self._params.num_labels, activation='softmax', name='softmax_out')(z)

        model = Model(inputs=inputs, outputs=pred)
        model.summary(print_fn=lambda x: logging.info(x + '\n'))
        model.compile(loss=loss, optimizer=self._params.optimizer)

        self.model = model

    def build_bert(self, verbose=True):
        """
        build bert + crf model for sequence model
        """
        # bert inputs
        bert_word_ids = Input(batch_shape=(None, self._params.max_sent_len), dtype="int32", name="bert_word_input")
        bert_mask_ids = Input(batch_shape=(None, self._params.max_sent_len), dtype="int32", name='bert_mask_input')
        bert_segment_ids = Input(batch_shape=(None, self._params.max_sent_len), dtype="int32", name="bert_segment_input")
        
        inputs = [bert_word_ids, bert_mask_ids, bert_segment_ids]

        bert_out = BertLayer(n_fine_tune_layers=self._params.n_fine_tune_layers, bert_path=self._params.bert_path, name="bert_layer")([bert_word_ids, bert_mask_ids, bert_segment_ids])

        features = bert_out

        if self._params.use_dict:
            if verbose: logging.info("use user dict features")
            dict_ids = Input(batch_shape=(None, self._params.max_sent_len), dtype='int32', name='dict_input')
            inputs.append(dict_ids)

            dict_embeddings = Embedding(input_dim=self._params.dict_vocab_size,
                                        output_dim=self._params.dict_embedding_dim,
                                        mask_zero=True,
                                        name='dict_embedding')(dict_ids)

            features = Concatenate(name="bert_and_dict_features")([features, dict_embeddings])

        z = Dense(self._params.fc_dim, activation='relu', name="fc_dense")(features)

        if self._params.use_crf:
            if verbose: logging.info('use crf decode layer')
            crf = CRF(self._params.num_labels, sparse_target=False,
                        learn_mode='marginal', test_mode='marginal', name='crf_out')
            loss = crf.loss_function
            pred = crf(z)
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self._params.num_labels, activation='softmax', name='softmax_out')(z)

        model = Model(inputs=inputs, outputs=pred)
        model.summary(print_fn=lambda x: logging.info(x + '\n'))

        # It is recommended that you use this optimizer for fine tuning, since this
        # is how the model was trained (note that the Adam m/v variables are NOT
        # loaded from init_checkpoint.)
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=1e-5,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        
        model.compile(loss=loss, optimizer=optimizer)

        self.model = model


    def validate_data(self, data, check_pos=True, check_label=True):
        """
        validate input data
        """
        if 'word' not in data:
            raise ValueError('data must contains word sequences')
        if check_pos:
            if 'pos' not in data:
                raise ValueError('data must contains pos sequences')
        if check_label:
            if 'label' not in data:
                raise ValueError('data must contains label sequences')

    def fit(self, train_data, valid_data, verbose=True):
        """
        model training
        """

        if self.model is None:
            raise ValueError("please build or restore your model first.")
 

        check_pos = self._params.dict.get('use_pos', False)
        self.validate_data(train_data, check_pos=check_pos, check_label=True)
        self.validate_data(valid_data, check_pos=check_pos, check_label=True)
        
        train_iter = DataIterator(train_data, batch_size=self._params.batch_size, input_processor=self.input_processor, shuffle=True)
        valid_iter = DataIterator(valid_data, batch_size=self._params.batch_size, input_processor=self.input_processor, shuffle=False)

        #valid_data = self.indexer.transform(valid_data, params, mode="evaluate")
        f1 = F1score(valid_iter, input_processor=self.input_processor)
        cb_early = EarlyStopping(monitor='f1', min_delta=0, mode='max',
                                    patience=self._params.early_stop, verbose=1, restore_best_weights=True)
        
        callbacks = [f1, cb_early]
        self.model.fit_generator(generator=train_iter,
                                    epochs=self._params.max_train_epoch,
                                    callbacks=callbacks,
                                    shuffle=True,
                                    verbose=verbose)

    def predict(self, data):
        """
        model predict
        """
        check_pos = self._params.dict.get('use_pos', False)
        self.validate_data(data, check_pos=check_pos, check_label=False)
        x_inputs = self.input_processor.transform(data, mode="predict")
        y_pred = self.model.predict_on_batch(x_inputs)

        y_pred, y_prob = self.input_processor.inverse_transform(y_pred, out_prob=True)

        if self._params.model == 'bert':
            mapping_seqs = data['bert_mapping']
            new_y_pred, new_y_prob = [], []
            for i, mapping_seq in enumerate(mapping_seqs):
                pred_seq, prob_seq = [], []
                for j, mapping_i in enumerate(mapping_seq):
                    if mapping_i not in {"[CLS]", "[SEP]", "X"}:
                        pred_seq.append(y_pred[i][j])
                        prob_seq.append(y_prob[i][j])

                new_y_pred.append(pred_seq)
                new_y_prob.append(prob_seq)

            return new_y_pred, new_y_prob

        return y_pred, y_prob

    def evaluate(self, data):
        """
        model evaluate

        Args:
            data: {'word': [], 'pos':[], 'size': 0}
        """
        check_pos = self._params.dict.get('use_pos', False)
        self.validate_data(data, check_pos=check_pos, check_label=True)
        data_iter = DataIterator(data, batch_size=self._params.batch_size, input_processor=self.input_processor, shuffle=False)

        labels = []
        predicts = []

        for i in range(len(data_iter)):
            x, y_true_seqs = data_iter[i]
            y_pred_seqs = self.model.predict_on_batch(x)

            y_true_seqs = self.input_processor.inverse_transform(y_true_seqs)
            y_pred_seqs = self.input_processor.inverse_transform(y_pred_seqs)

            if self.input_processor.name == "bert_input_processor":
                new_y_true_seqs, new_y_pred_seqs = [], []
                for y_true_seq, y_pred_seq in zip(y_true_seqs, y_pred_seqs):
                    new_y_true_seq, new_y_pred_seq = [], []
                    for true_tag, pred_tag in zip(y_true_seq, y_pred_seq):
                        if true_tag not in {"[CLS]", "[SEP]", "X"}:
                            new_y_true_seq.append(true_tag)
                            new_y_pred_seq.append(pred_tag)

                    new_y_true_seqs.append(new_y_true_seq)
                    new_y_pred_seqs.append(new_y_pred_seq)

                y_true_seqs = new_y_true_seqs
                y_pred_seqs = new_y_pred_seqs

            labels.extend(y_true_seqs)
            predicts.extend(y_pred_seqs)

        score = f1_score(labels, predicts)
        logging.info(' - f1: {:04.2f}'.format(score * 100))
        logging.info(classification_report(labels, predicts))

        return score

    def restore(self, weights_file):
        """
        restore model from local
        """

        self.model.load_weights(weights_file, by_name=True)

    def save(self, weights_file):
        """
        save model to model_path
        """
        
        self.model.save_weights(weights_file)

    def export_sm(self, tf_sm_dir):
        """
        export model as Saved Model format for online serving
        :param tf_sm_dir: output fold
        :return:
        """
        tf.saved_model.simple_save(
            K.get_session(),
            tf_sm_dir,
            inputs={i.name: i for i in self.model.inputs},
            outputs={out.name: out for out in self.model.outputs}
        )

