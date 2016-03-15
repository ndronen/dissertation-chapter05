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

import sys
import collections
import json
import numpy as np
import pandas as pd

import importlib

import modeling.utils
from spelling.mitton import evaluate_ranks
from spelling.utils import build_progressbar as build_pbar
import spelling.preprocess

import chapter05.data
import chapter05.dataset

#import models.binary.model as M

def load_model_module(model_dir):
    sys.path.append('.')
    model_module_path = model_dir.replace('/', '.') + '.model'
    return importlib.import_module(model_module_path)

def run_analysis(mode, model_dir='models/binary/d8bf93f8e71111e5a584fcaa149e39ea', csv_path='../chapter04/data/train.csv', random_state=17, model_weights='model.h5', ranks=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500]):
    # Load the model configuration, then load the entire data set and split
    # it into train, validation, and test.
    config = load_config(model_dir)

    M = load_model_module(model_dir)

    df = M.load_data(csv_path, config['min_frequency'],
            config['min_length'], config['max_length'])

    df_train, df_valid, df_test = M.split_data(df, random_state=config['seed'])

    # Get the vocabulary from the entire data set, then delete it, so
    # we don't mistakenly use it below.
    vocabulary = df.real_word.unique().tolist()
    del df

    df = {
            'train': df_train,
            'valid': df_valid,
            'test': df_test
        }[mode]

    # Load the suggestions for all examples (train, validation, and test).
    df_sugg_all = load_suggestions()

    # Build a dict of lists that maps each non-word in df_sugg_all to
    # the indices of the rows for that non-word in df_sugg_all.
    non_word_index = build_suggestions_index(df_sugg_all)

    # Before taking a subset of the suggestions, add a column indicating
    # whether the correct word is in the suggestions.
    add_correct_word_is_in_suggestions(df_sugg_all, non_word_index)

    # Since we will only be doing the analysis on a subset of the
    # examples, get the indices of the suggestions of just those examples
    # and use them to get a data frame of suggestions corresponding to
    # only those examples.
    indices = get_suggestion_indices_for_mode(df, non_word_index)
    df_sugg = df_sugg_all.iloc[indices, :].copy()

    # Format and sort the dictionary suggestions.
    df_sugg_fmt = format_dict_suggs_for_analysis(df, df_sugg, non_word_index)
    df_sugg_fmt.sort_values(['non_word', 'correct_word', 'suggestion'], inplace=True)
    df_sugg_fmt.drop_duplicates(['non_word', 'correct_word', 'suggestion'], inplace=True)

    # Load the model and run the data through it.
    config['model_weights'] = model_dir + '/' + model_weights
    model_cfg = modeling.utils.ModelConfig(**config)
    model = M.build_model(model_cfg)

    # Each example consists of a non-word, a candidate real word,
    # and the true correction.  

    examples = build_examples(
            df_sugg_fmt.word.tolist(),
            df_sugg_fmt.real_word.tolist(),
            df_sugg_fmt.correct_word.tolist())

    results = run_model(model, model_cfg, examples)

    df_model_fmt = format_model_suggs_for_analysis(results)
    df_model_fmt.sort_values(['non_word', 'correct_word', 'suggestion'], inplace=True)

    dictionaries = {}
    dictionaries['Aspell with Jaro-Winkler'] = df_sugg_fmt
    dictionaries['ConvNet (binary)'] = df_model_fmt

    ranks_df = compute_ranks(dictionaries, ranks)
    ranks_df['Candidates'] = 'All'
    ranks_df_no_near_miss = compute_ranks_no_near_miss(dictionaries, ranks)
    ranks_df_no_near_miss['Candidates'] = 'No near-miss'

    all_ranks_df = pd.concat([ranks_df, ranks_df_no_near_miss], axis=0)
    all_ranks_df['Mode'] = mode

    all_ranks_df.to_csv('ranks-%s.csv' % mode, sep='\t', encoding='utf8')
    print('wrote ranks-%s.csv' % mode)

    return {
            'df': df,
            'df_sugg': df_sugg_fmt,
            'df_model': df_model_fmt,
            'ranks': all_ranks_df,
            'model': model,
            'model_cfg': model_cfg
            }

def run_model(model, model_cfg, examples):
    test_results = collections.defaultdict(list)
    pbar = build_pbar(examples)
    for i,non_word in enumerate(examples.keys()):
        pbar.update(i+1)
        candidates = [c[0] for c in examples[non_word]]
        true_corrections = [c[1] for c in examples[non_word]]
        data_dict = build_model_input(non_word, candidates, model_cfg)
        rank = rank_candidates(model, data_dict, model_cfg.target_name)
        for j,r in enumerate(rank):
            test_results[non_word].append((candidates[j], r, true_corrections[j]))
    pbar.finish()
    return test_results

def build_examples(non_words, real_words, correct_words):
    pbar = build_pbar(non_words)
    examples = collections.defaultdict(list)
    for i,non_word in enumerate(non_words):
        pbar.update(i+1)
        examples[non_word].append(
                (real_words[i], correct_words[i]))
    pbar.finish()
    return examples

def rank_candidates(model, example, target_name):
    # Return the order of the maximum probabilities, from greatest to least.
    prob = model.predict(example)[target_name]
    # Reverse the argsort.
    temp = prob[:, 1].argsort()[::-1]
    rank = np.empty(len(prob), int)
    rank[temp] = np.arange(len(prob))
    assert min(rank) == 0
    assert max(rank) < len(prob)
    return rank

def build_model_input(non_word, candidates, config):
    # Build a character matrix of the non-word inputs, first marking
    # the non-words with start and end of token characters.
    non_word_inputs = ['^'+non_word+'$' for candidate in candidates]
    non_word_matrix, non_word_kept = spelling.preprocess.build_char_matrix(
            [non_word] * len(candidates), width=config.model_input_width)

    # Build a character matrix of the real word inputs, first marking
    # the real words with start and end of token characters.
    candidate_word_inputs = ['^'+candidate+'$' for candidate in candidates]
    candidate_word_matrix, candidate_word_kept = spelling.preprocess.build_char_matrix(
        candidates, width=config.model_input_width)

    mask = non_word_kept & candidate_word_kept
    assert sum(~mask) == 0
    idx = np.where(non_word_kept & candidate_word_kept)[0]
    assert len(idx) == len(candidates)

    non_word_matrix = non_word_matrix[mask]
    candidate_word_matrix = candidate_word_matrix[mask]

    return {
        'non_word_input': non_word_matrix[mask],
        'candidate_word_input': candidate_word_matrix[mask]
        }

def format_model_suggs_for_analysis(test_results):
    #    - non_word                         <- keys of test_results
    #    - suggestion                       <- first element of values of test_results
    #    - suggestion_index                 <- second element of values of test_results
    #    - correct_word                     <- third element of values of test_results
    #    - correct_word_is_in_suggestions   <- compare first and third elements of every value for a key
    #    - correct_word_in_dict             <- this is always 1, since we're using the Aspell vocabulary
    model_suggs = []

    for non_word in test_results.keys():
        d = {}
        d['non_word'] = non_word
        d['correct_word_in_dict'] = 1

        sugg_tuples = test_results[non_word]
        d['correct_word_is_in_suggestions'] = int(any([t[0] == t[2] for t in sugg_tuples]))

        for sugg_tuple in sugg_tuples:
            d['suggestion'] = sugg_tuple[0]
            d['suggestion_index'] = sugg_tuple[1]
            d['correct_word'] = sugg_tuple[2]
            model_suggs.append(dict(d))

    return pd.DataFrame(data=model_suggs)

def format_dict_suggs_for_analysis(df, df_sugg, non_word_index):
    # TODO: the following columns must be in df_sugg before we can
    # call spelling.mitton.evaluate_ranks.
    #
    #    - non_word                         <- word
    #    - suggestion                       <- real_word
    #    - suggestion_index                 <- get this from df_sugg['rank']
    #    - correct_word                     <- get this from df['real_word']
    #    - correct_word_is_in_suggestions   <- this is added before df_sugg is passed to this function
    #    - correct_word_in_dict             <- this is always 1, since we're using the Aspell vocabulary

    df_sugg['suggestion'] = df_sugg['real_word']
    df_sugg['non_word'] = df_sugg['word']
    df_sugg['suggestion_index'] = df_sugg['rank']

    corrections = dict(zip(df.word, df.real_word))
    df_sugg['correct_word'] = [corrections[word] for word in df_sugg.non_word]

    #non_words_with_true_correction_in_candidate_list = set(
    #        df_sugg[df_sugg.binary_correction_target == 1].non_word)
    #correct_word_is_in_suggestions = []

    """
    pbar = build_pbar(df_sugg.non_word)
    for i,non_word in enumerate(df_sugg.non_word):
        pbar.update(i+1)
        #if non_word in non_words_with_true_correction_in_candidate_list:
        if is_correct_word_in_suggestions(non_word, df_sugg, non_word_index):
            correct_word_is_in_suggestions.append(1)
        else:
            correct_word_is_in_suggestions.append(0)
    pbar.finish()
    """

    #df_sugg['correct_word_is_in_suggestions'] = correct_word_is_in_suggestions

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

def get_suggestion_indices_for_mode(df, non_word_index):
    pbar = build_pbar(df.word)
    indices = []
    for i,non_word in enumerate(df.word):
        pbar.update(i+1)
        indices.extend(non_word_index[non_word])
    pbar.finish()
    return indices

def add_correct_word_is_in_suggestions(df_sugg, non_word_index):
    # The non-word column in df_sugg is just called `word`.
    non_words = df_sugg.word.unique().tolist()
    is_in_suggestions = {}
    pbar = build_pbar(non_words)
    for i,non_word in enumerate(non_words):
        pbar.update(i+1)
        indices = non_word_index[non_word]
        targets = df_sugg.iloc[indices, :].binary_correction_target
        is_in_suggestions[non_word] = np.any(targets == 1).astype(int)
    pbar.finish()

    df_sugg['correct_word_is_in_suggestions'] = \
        df_sugg.word.apply(lambda nw: is_in_suggestions[nw])

def contains_space_or_hyphen(word):
    return any([c in word for c in ["-", " "]])

def remove_split_and_hyphenated_suggestions(df_sugg):
    df_sugg = df_sugg.copy()

    df_sugg.suggestion_index = df_sugg.suggestion_index.astype(np.float)

    # Find the groups first, before setting the suggestion
    # index (i.e. the rank) of the space- or hyphen-containing
    # suggestions to NaN.
    idx = np.where(df_sugg.suggestion_index == 0)[0]
    offsets = [idx[i:i+2] for i in range(len(idx))]
    offsets[-1] = [offsets[-1].item(), len(df_sugg)]

    # Now set the suggestion index of the space- or hyphen-containing
    # suggestions to NaN.
    exclude_from_candidate_list = df_sugg.suggestion.apply(
            contains_space_or_hyphen)
    mask_idx = np.where(exclude_from_candidate_list)[0]
    masked_sugg_idx = df_sugg.suggestion_index.values
    masked_sugg_idx[mask_idx] = np.nan
    df_sugg.suggestion_index = masked_sugg_idx

    groups = []

    pbar = build_pbar(offsets)

    for i,(start,end) in enumerate(offsets):
        pbar.update(i+1)
        df_tmp = df_sugg.iloc[start:end, :].copy()
        excluded = df_tmp.suggestion_index.isnull()
        if any(excluded):
            n = sum(~excluded)
            sugg_idx = df_tmp.suggestion_index.values
            sugg_idx[~np.isnan(sugg_idx)] = np.arange(n)
            df_tmp.suggestion_index = sugg_idx
        groups.append(df_tmp)

    pbar.finish()

    return pd.concat(groups, axis=0)


def compute_ranks_no_near_miss(dictionaries, ranks):
    """
    Exclude from the dictionary candidate list any word with a space or
    a hyphen in it.  This removes some of the bias the results we report
    using generated corpora of errors, which by design never have a space
    or a hyphen in the true corrections.
    """
    d = {}
    for name,df in dictionaries.items():
        df = df.copy()
        d[name] = remove_split_and_hyphenated_suggestions(df)
    return compute_ranks(d, ranks)

def compute_ranks(dictionaries, ranks):
    ranks_df = evaluate_ranks(dictionaries, ranks=ranks)
    ranks_df['Non-word Length'] = -1
    ranks_df_by_length = compute_accuracy_at_k_by_non_word_length(
            dictionaries, ranks)
    return pd.concat([ranks_df, ranks_df_by_length], axis=0)

def compute_accuracy_at_k_by_non_word_length(dictionaries, ranks):
    d = {}
    start = 1000
    end = 0
    for name,df in dictionaries.items():
        df = df.copy()
        df['non_word_len'] = df.non_word.apply(len)
        start = min(df.non_word_len.min(), start)
        end = max(end, df.non_word_len.max())
        d[name] = df

    rank_dfs = []
    for length in range(start, end+1):
        d_word_len = {}
        for name,df in d.items():
            d_word_len[name] = df[df.non_word_len == length]
        rank_df = spelling.mitton.evaluate_ranks(d_word_len, ranks=ranks)
        rank_df['Non-word Length'] = length
        rank_dfs.append(rank_df)

    return pd.concat(rank_dfs, axis=0)
