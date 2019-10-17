"""EVALUATION OF NATURALNESS

This code can be used to evaluate the naturalness of output sentiment texts of examined style transfer models.

For a baseline understanding of what is considered "natural," any method used for automated evaluation of naturalness
also requires an understanding of the human-sourced input texts.

Inspired by the adversarial evaluation approach in "Generating Sentences from a Continuous Space"
(Bowman et al., 2016), we trained unigram logistic regression classifiers and LSTM logistic regression classifiers
on samples of input texts and output texts for each style transfer model.

Via adversarial evaluation, the classifiers must distinguish human-generated inputs from machine-generated outputs.
The more natural an output is, the likelier it is to fool an adversarial classifier.

We calculate percent agreement with human judgments. Both classifiers show greater agreement on which texts are
considered more natural with humans given relative scoring tasks than with those given absolute scoring tasks.

Usage:
    - View percent agreement between automated scores and human scores              -> display_agreements()
    - Calculate naturalness scores for texts with clf, a NaturalnessClassifier      -> clf.score(...)

You can find examples of more detailed usage commands below.

"""

from keras.models import load_model as load_keras_model
from keras.preprocessing.sequence import pad_sequences
from tokenizer import RE_PATTERN
import numpy as np
import re

ASPECT = 'naturalness'
AUTOMATED_EVALUATION_BASE_PATH = '../evaluations/automated/{ASPECT}/sentence_level'
CLASSIFIER_BASE_PATH = '../models/naturalness_classifiers'
MAX_SEQ_LEN = 30  # for neural classifier
TEXT_VECTORIZER = load_keras_model('../models/vectorizer.pkl')

# adjust vocabulary to account for unknowns
VOCABULARY = TEXT_VECTORIZER.vocabulary_
VOCABULARY['CUSTOM_UNKNOWN'] = len(VOCABULARY)+1


# DATA PREP
def convert_to_indices(text):
    # tokenize input text
    tokens = re.compile(RE_PATTERN).split(text)
    non_empty_tokens = list(filter(lambda token: token, tokens))

    indices = []

    # collect indices of tokens in vocabulary
    for token in non_empty_tokens:
        if token in VOCABULARY:
            index = VOCABULARY[token]
        else:
            index = VOCABULARY['CUSTOM_UNKNOWN']

        indices.append(index)

    return indices


def format_inputs(texts):
    # prepare texts for use in neural classifier
    texts_as_indices = []
    for text in texts:
        texts_as_indices.append(convert_to_indices(text))
    return pad_sequences(texts_as_indices, maxlen=MAX_SEQ_LEN, padding='post', truncating='post', value=0.)


# NATURALNESS CLASSIFIERS
class NaturalnessClassifier:
    '''
    An external classifier was trained for each examined style transfer model -
    more specifically using its inputs and outputs, of course excluding test samples.

    Use UnigramBasedClassifier or NeuralBasedClassifier to load a
    trained classifier and score texts of a given style transfer model.
    The scores represent the probabilities of the texts being 'natural'.

    '''

    pass


class UnigramBasedClassifier(NaturalnessClassifier):
    def __init__(self, style_transfer_model_name, text_vectorizer=TEXT_VECTORIZER):
        self.path = '{CLASSIFIER_BASE_PATH}/unigram_{style_transfer_model_name}.pkl'
        self.classifier = load_keras_model(self.path)
        self.text_vectorizer = text_vectorizer

    def score(self, texts):
        vectorized_texts = self.text_vectorizer.transform(texts)
        distribution = self.classifier.predict_proba(vectorized_texts)
        scores = distribution[:, 1]  # column 1 represents probability of being 'natural'
        return scores


class NeuralBasedClassifier(NaturalnessClassifier):
    def __init__(self, style_transfer_model_name):
        self.path = '{CLASSIFIER_BASE_PATH}/neural_{style_transfer_model_name}.h5'
        self.classifier = load_keras_model(self.path)

    def score(self, texts):
        inps = format_inputs(texts)
        distribution = self.classifier.predict(inps)
        scores = distribution.squeeze()
        return scores


# CALCULATION OF AGREEMENTS
def generate_judgments(input_text_scores, output_text_scores):
    '''
    Compare naturalness scores of input and output texts, representing
    the case where an input is scored as more natural with 1, output with 0,
    and neither with None. Generate "judgments" with such labels.

    Parameters
    ----------
    input_text_scores : numpy.ndarray
        Naturalness scores assigned to input texts
    output_text_scores : numpy.ndarray
        Naturalness scores assigned to output texts

    Returns
    -------
    judgments : numpy.ndarray
        Labels representing which texts were marked as more natural

    '''

    judgments = []

    for i in range(len(input_text_scores)):
        input_text_score = input_text_scores[i]
        output_text_score = output_text_scores[i]

        if input_text_score != output_text_score:
            # represent input text being scored as more natural as 1, otherwise 0
            val = int(input_text_score > output_text_score)
        else:
            val = None
        judgments.append(val)

    return np.array(judgments)


def format_relative_judgments(judgments):
    '''
    Raters provided judgments of which of a given input ('A') and
    output text ('B') is more natural. Represent 'A' as 1 and 'B' as 0
    for downstream comparison with judgments from other scoring methods
    that use this representation.

    Parameters
    ----------
    judgments : list
        List of 'A', 'B', and/or None judgments

    Returns
    -------
    List of formatted judgments

    '''

    judgments_map = {'A': 1, 'B': 0, None: None}
    return list(map(lambda judgment: judgments_map[judgment], judgments))


# EXAMPLE USAGE (uncomment the following to get naturalness scores with a trained adversarial classifier)

# model = 'CAAE'
# param = MODEL_TO_PARAM_NAMES[model]
# val = get_val_as_str(0.1)

# # load data
# negative_to_positive_transfers = load_dataset(f'../transfer_model_outputs/{model}/{param}_{val}/sentiment.test.0.tsf')
# positive_to_negative_transfers = load_dataset(f'../transfer_model_outputs/{model}/{param}_{val}/sentiment.test.1.tsf')
# output_texts = merge_datasets(negative_to_positive_transfers, positive_to_negative_transfers)

# # score
# neural_classifier = NeuralBasedClassifier(model)
# print(neural_classifier.score(output_texts))
