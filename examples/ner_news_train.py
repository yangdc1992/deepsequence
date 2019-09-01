"""
Example from training to saving.
"""
import argparse
import logging

from context import deepsequence
from deepsequence.dataset import load_conll
from deepsequence.model import SequenceModel
from deepsequence.config import Params
from deepsequence.utils import set_logger


def main(params):

    train_file = params.data_dir + '/train.txt'
    valid_file = params.data_dir + '/valid.txt'
    train_data = load_conll(train_file, params)
    valid_data = load_conll(valid_file, params)

    model = SequenceModel(params)

    weights_file = params.data_dir + '/model/weights.h5'

    if params.continue_previous_training is True:

        logging.info("restore model from local")
        model.restore(weights_file)

    model.fit(train_data, valid_data, verbose=True)
    model.evaluate(valid_data)
    logging.info("model weights save to {}".format(weights_file))
    model.save(weights_file)

    tf_saved_model_dir = params.data_dir + '/model/tf_saved_model'
    model.export_sm(tf_saved_model_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''train deep sequence model''')
    parser.add_argument('--config', required=True)

    set_logger('train.log')

    args = parser.parse_args()

    logging.info("parse config file path: {}".format(args.config))

    params = Params(args.config)
    logging.info("parameters: {}".format(params.dict))

    try:
        main(params)
    except Exception as e:
        logging.error('run fail', exc_info=True)