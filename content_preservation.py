"""EVALUATION OF CONTENT PRESERVATION

This code can be used for evaluation of content preservation between input and output sentiment texts of a style transfer model.

Word Mover's Distance (WMD) on texts with style masking (i.e. placeholders used in place of style words)
exhibited the highest correlation with human evaluations of the same texts.

Usage:
    - Mask style words in a set of texts prior to evaluation                -> mark_style_words(texts, mask_style=True)
    - Train a Word2Vec model for your dataset, for use in WMD calculation   -> train_word2vec_model(...)
    - Calculate WMD scores for your own input/output texts                  -> calculate_wmd_scores(...)

You can find examples of more detailed usage commands below.

"""

from gensim.models.word2vec import Word2Vec
from tokenizer import tokenize


ASPECT = 'content_preservation'
AUTOMATED_SCORES_PATH = '../evaluations/automated/content_preservation/sentence_level'
CUSTOM_STYLE = 'customstyle'
STYLE_MODIFICATION_SETTINGS = ['style_masked', 'style_removed']


# DATA PREP
def mark_style_words(texts, style_tokens=None, mask_style=False):
    '''
    Mask or remove style words (based on a set of style tokens) from input texts.

    Parameters
    ----------
    texts : list
        String inputs
    style_tokens : set
        Style tokens
    mask_style : boolean
        Set to False to remove style tokens, True to replace with placeholder

    Returns
    -------
    edited_texts : list
        Texts with style tokens masked or removed

    '''

    edited_texts = []

    for text in texts:
        tokens = tokenize(text)
        edited_tokens = []

        for token in tokens:
            if token.lower() in style_tokens:
                if mask_style:
                    edited_tokens.append(CUSTOM_STYLE)
            else:
                edited_tokens.append(token)

        edited_texts.append(' '.join(edited_tokens))

    return edited_texts


def generate_style_modified_texts(texts):

    # ensure consistent tokenization under different style modification settings
    unmasked_texts = mark_style_words(texts, {})
    texts_with_style_removed = mark_style_words(texts)
    texts_with_style_masked = mark_style_words(texts, mask_style=True)
    return unmasked_texts, texts_with_style_removed, texts_with_style_masked


# MODELS / SCORING OF WMD
def train_word2vec_model(texts, path):
    tokenized_texts = []
    for text in texts:
        tokenized_texts.append(tokenize(text))
    model = Word2Vec(tokenized_texts)
    model.save(path)


def load_word2vec_model(path):
    model = Word2Vec.load(path)
    model.init_sims(replace=True)  # normalize vectors
    return model


def calculate_wmd_scores(references, candidates, wmd_model):
    '''
    Calculate Word Mover's Distance for each (reference, candidate)
    pair in a list of reference texts and candidate texts.

    The lower the distance, the more similar the texts are.

    Parameters
    ----------
    references : list
        Input texts
    candidates : list
        Output texts (e.g. from a style transfer model)
    wmd_model : gensim.models.word2vec.Word2Vec
        Trained Word2Vec model

    Returns
    -------
    wmd_scores : list
        WMD scores for all pairs

    '''

    wmd_scores = []

    for i in range(len(references)):
        wmd = wmd_model.wv.wmdistance(tokenize(references[i]), tokenize(candidates[i]))
        wmd_scores.append(wmd)

    return wmd_scores


# EXAMPLE USAGE (uncomment the following to play around with code)
# load data to train models used for WMD calculations
# all_texts = load_dataset('../data/sentiment.all')
# all_texts_style_masked = mark_style_words(all_texts, mask_style=True)

# # train models
# w2v_model_path = '../models/word2vec_unmasked'
# w2v_model_style_masked_path = '../models/word2vec_masked'
# # train_word2vec_model(all_texts, w2v_model_path)
# # train_word2vec_model(all_texts_style_masked, w2v_model_style_masked_path)
# w2v_model = load_word2vec_model(w2v_model_path)
# w2v_model_style_masked = load_word2vec_model(w2v_model_style_masked_path)

# # load texts under different style modification settings
# input_neg_texts = load_dataset('../data/sentiment.test.0')
# input_pos_texts = load_dataset('../data/sentiment.test.1')
# input_texts = merge_datasets(input_neg_texts, input_pos_texts)
# unmasked_inputs, inputs_with_style_removed, inputs_with_style_masked = generate_style_modified_texts(input_texts)
