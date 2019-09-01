"""
Custom callbacks.
"""
import logging

import numpy as np
from keras.callbacks import Callback
from deepsequence.seqeval import f1_score, classification_report
from deepsequence.utils import get_lengths


class F1score(Callback):

    def __init__(self, data, input_processor):
        super(F1score, self).__init__()
        self.data = data
        self.input_processor = input_processor

    def on_epoch_end(self, epoch, logs={}):

        labels = []
        predicts = []

        for i in range(len(self.data)):
            x, y_true_seqs = self.data[i]
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
        logs['f1'] = score