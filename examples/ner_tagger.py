import spacy

from context import deepsequence
from deepsequence.model import SequenceModel
from deepsequence.config import Params

class Tagger(object):
    def __init__(self, weights_file, params):
        self.params = params
        self.tokenizer = spacy.load('en_core_web_sm')
        self.model = SequenceModel(params)
        self.model.restore(weights_file)

    def analysis(self, text):
        spacy_tokens = self.tokenizer(text)
        print(spacy_tokens)
        data = {}
        data['word'] = [[token.text for token in spacy_tokens]]
        if self.params.dict.get("use_pos", False):
            data['pos'] = [[token.pos_ for token in spacy_tokens]]

        data['size'] = 1
        predicts, probs = self.model.predict(data)
        predicts, probs = predicts[0], probs[0]
        print(predicts)

        print([(token.text, tag, prob) for token, tag, prob in zip(spacy_tokens, predicts, probs)])



if __name__ == "__main__":
    params = Params('/home/decheng/project/deepsequence/examples/bilstm_parameters.json')
    weights_file = params.data_dir + '/model/weights.h5'
    tagger = Tagger(weights_file, params)
    tagger.analysis('I live in London')
