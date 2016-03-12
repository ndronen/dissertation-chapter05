# Having trained the binary isolated non-word error correction models,
# I now need to analyze the results.   An overview of the procedure is:
#
# * Take a subset of the test set of generated errors.  The subset
#   should consist of 2 examples of each word.  That should yield
#   about 250k examples.  For each non-word, obtain its candidate
#   from the Aspell and edit distance retrievers (with both of them
#   sorted by Jaro-Winkler distance).  From these, build a data frame
#   with the following columns:
#   
#    - non_word
#    - dictionary
#    - suggestion 
#    - suggestion_index
#    - correct_word
#    - correct_word_is_in_suggestions
#    - correct_word_in_dict
#
# * Rank each of the candidate lists using the binary model.  Save the
#   ranks to the CSV file.
#
# * Report top-k accuracy for each model for k = 1,2,3,4,5,10,20,30,40,50
#   using spelling.mitton.evaluate_ranks.
#
# * Repeat the analysis using the Mitton corpora.

import collections
import json
import numpy as np
import pandas as pd

import modeling.utils
from spelling.mitton import evaluate_ranks
from spelling.utils import build_progressbar as build_pbar

import models.binary.model as M
import chapter05.data

def run(model_dir='models/binary/train_binary_crossval_01/cos_10_100_10_000', csv_path='../chapter04/data/train.csv', random_state=17):
    df_test, df_test_sugg = run_dictionaries_on_generated_examples(
            model_dir, csv_path, random_state)
    non_word_index = build_suggestions_index(df_test_sugg)
    df_sugg_formatted = format_for_evaluation(df_test_sugg, df_test, non_word_index)
    return evaluate_ranks({ 'AspellJaroWinkler': df_sugg_formatted },
        ranks=[1,2,3,4,5,10,20,30,40,50])

def run_dictionaries_on_generated_examples(model_dir, csv_path='../chapter04/data/train.csv', random_state=17):
    """
    Load the generated examples, get the test set, 
    """
    config = load_config(model_dir)

    df = M.load_data(csv_path, config['min_frequency'],
            config['min_length'], config['max_length'])

    df_train, df_valid, df_test = M.split_data(df, random_state=config['seed'])

    df_sugg = load_suggestions()
    non_word_index = build_suggestions_index(df_sugg)
    indices = get_test_set_suggestion_indices(df_test, non_word_index)
    df_test_sugg = df_sugg.iloc[indices, :].copy()

    return df_test, df_test_sugg

def format_for_evaluation(df_sugg, df_test, non_word_index):
    # TODO: the following columns must be in df_test_sugg before we can
    # call spelling.mitton.evaluate_ranks.
    #
    #    - non_word                         <- word
    #    - suggestion                       <- real_word
    #    - suggestion_index                 <- get this from df_sugg['rank']
    #    - correct_word                     <- get this from df['real_word']
    #    - correct_word_is_in_suggestions   <- determine this by whether binary_suggestion_target is 1 for a word
    #    - correct_word_in_dict             <- this is always 1, since we're using the Aspell vocabulary

    df_sugg['suggestion'] = df_sugg['real_word']
    df_sugg['non_word'] = df_sugg['word']
    df_sugg['suggestion_index'] = df_sugg['rank']

    corrections = dict(zip(df_test.word, df_test.real_word))
    df_sugg['correct_word'] = [corrections[word] for word in df_sugg.non_word]

    non_words_with_true_correction_in_candidate_list = set(
            df_sugg[df_sugg.binary_correction_target == 1].non_word)
    correct_word_is_in_suggestions = []

    pbar = build_pbar(df_sugg.non_word)
    for i,non_word in enumerate(df_sugg.non_word):
        pbar.update(i+1)
        if non_word in non_words_with_true_correction_in_candidate_list:
            correct_word_is_in_suggestions.append(1)
        else:
            correct_word_is_in_suggestions.append(0)
    pbar.finish()

    df_sugg['correct_word_is_in_suggestions'] = correct_word_is_in_suggestions

    df_sugg['correct_word_in_dict'] = 1

    return df_sugg

def load_suggestions(suggestions_file='/localwork/ndronen/spelling/generated-error-suggestions.csv.gz'):
    df_sugg = pd.read_csv(suggestions_file, sep='\t', encoding='utf8')
    print(df_sugg.shape)
    return df_sugg

def build_suggestions_index(df_sugg):
    non_word_index = collections.defaultdict(list)
    for i,non_word in enumerate(df_sugg.word):
        non_word_index[non_word].append(i)
    return non_word_index 

def load_config(model_dir):
    return json.load(open(model_dir + '/config.json'))

def get_test_set_suggestion_indices(df_test, non_word_index):
    pbar = build_pbar(df_test.word)
    indices = []
    for i,non_word in enumerate(df_test.word):
        pbar.update(i+1)
        indices.extend(non_word_index[non_word])
    pbar.finish()
    return indices

def is_correct_word_in_suggestions(non_word, df_sugg, non_word_index):
    indices = non_word_index[non_word]
    targets = df_sugg.iloc[indices, :].binary_correction_target
    return np.any(targets == 1)
