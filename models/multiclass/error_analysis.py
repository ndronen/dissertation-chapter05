# Having trained the multiclass isolated non-word error correction model,
# I now need to analyze the results.   An overview of the procedure is:
#
# * Take a subset of the test set of generated errors.  The subset
#   should consist of 2 examples of each real word.  That should yield
#   about 250k examples.  
# * The multiclass model should be evaluated both as a RETRIEVE
#   component and a RANK component.
# * To evaluate it as a RETRIEVE component, find the fraction of candidate
#   lists it outputs that contain the correct word.  Compare this to the
#   the fraction of candidate lists returned by Aspell and near-miss 
#   retrievers.
# * To evaluate it as a RANK component, take the ranked candidate lists
#   it outputs and compare them to the Jaro-Winkler and Random Forest
#   rankings.

import sys, os
import collections
import json
import pickle
import numpy as np
import pandas as pd

import importlib

import modeling.utils
from spelling.mitton import evaluate_ranks
import spelling.evaluate
import spelling.mitton
from spelling.utils import build_progressbar as build_pbar
from spelling.preprocess import build_char_matrix

import chapter05.data
import chapter05.dataset

BEST_MODEL_DIR = 'models/multiclass/b07b85ecf66f11e59ffbfcaa149e39ea'
RANKS = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500]
TRAIN_CSV_PATH = '../chapter04/data/train-real-words.csv'

def load_config(model_dir):
    return json.load(open(model_dir + '/config.json'))

def load_model_module(model_dir=BEST_MODEL_DIR):
    sys.path.append('.')
    model_module_path = model_dir.replace('/', '.') + '.model'
    return importlib.import_module(model_module_path)

def load_model(n_classes, model_dir=BEST_MODEL_DIR, model_weights='model.h5'):
    # Load the model and run the data through it.
    config = load_config(model_dir)
    config['model_weights'] = model_dir + '/' + model_weights
    model_cfg = modeling.utils.ModelConfig(**config)

    M = load_model_module(model_dir)
    model = M.build_model(model_cfg, n_classes)
    return model, model_cfg

def run_analysis(mode, model_dir=BEST_MODEL_DIR, csv_path=TRAIN_CSV_PATH, random_state=17, model_weights='model.h5', ranks=RANKS):
    # Load the model configuration, then load the entire data set and split
    # it into train, validation, and test.
    config = load_config(model_dir)
    model_cfg = modeling.utils.ModelConfig(**config)

    M = load_model_module(model_dir)

    df = M.load_data(model_cfg)
    vocabulary = M.build_vocabulary(df)
    df = M.take_non_word_examples(df)
    df = M.subset_non_words_by_length_and_freq(df, model_cfg)
    target_vocabulary = M.build_target_vocabulary(vocabulary, model_cfg)
    le = M.build_label_encoder(target_vocabulary)
    df_train, df_valid, df_test = M.split_data(df, random_state=model_cfg.seed)
    n_classes = len(target_vocabulary)

    retriever = M.build_retriever(vocabulary)

    model, _ = load_model(n_classes)

    return {
            'model': model,
            'model_cfg': model_cfg,
            'df_train': df_train,
            'df_valid': df_valid,
            'df_test': df_test,
            'label_encoder': le,
            'retriever': retriever
            }

    #df_sugg_all = load_suggestions()

    # Build a dict of lists that maps each non-word in df_sugg_all to
    # the indices of the rows for that non-word in df_sugg_all.
    #non_word_index = build_suggestions_index(df_sugg_all)

    # Before taking a subset of the suggestions, add a column indicating
    # whether the correct word is in the suggestions.
    #add_correct_word_is_in_suggestions(df_sugg_all, non_word_index)

    # Since we will only be doing the analysis on a subset of the
    # examples, get the indices of the suggestions of just those examples
    # and use them to get a data frame of suggestions corresponding to
    # only those examples.
    #indices = get_suggestion_indices_for_mode(df, non_word_index)
    #df_sugg = df_sugg_all.iloc[indices, :].copy()

    # Format and sort the dictionary suggestions.
    #df_sugg_fmt = format_dict_suggs_for_analysis(df, df_sugg, non_word_index)
    #df_sugg_fmt.sort_values(['non_word', 'correct_word', 'suggestion'], inplace=True)
    #df_sugg_fmt.drop_duplicates(['non_word', 'correct_word', 'suggestion'], inplace=True)

    # Load the model and run the data through it.
    #config['model_weights'] = model_dir + '/' + model_weights
    #model_cfg = modeling.utils.ModelConfig(**config)
    #model = M.build_model(model_cfg)

    # Each example consists of a non-word, a candidate real word,
    # and the true correction.  

    """
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
    """

def run_model(model, model_cfg, examples):
    test_results = collections.defaultdict(list)
    pbar = build_pbar(examples)
    for i,(non_word,correct_word) in enumerate(examples.keys()):
        pbar.update(i+1)
        candidates = examples[(non_word,correct_word)]
        #true_corrections = [c[1] for c in examples[non_word]]
        data_dict = build_model_input(non_word, candidates, model_cfg)
        rank = rank_candidates(model, data_dict, model_cfg.target_name)
        for j,r in enumerate(rank):
            test_results[(non_word,correct_word)].append((candidates[j], r))
    pbar.finish()
    return test_results

def build_examples(non_words, real_words, correct_words):
    """
    Input: lists of non-words, candidates, and corrections.
    Output: dict of (non-word,correction) keys, candidate list values.
    """
    pbar = build_pbar(non_words)
    examples = collections.defaultdict(list)
    for i,non_word in enumerate(non_words):
        pbar.update(i+1)
        correct_word = correct_words[i]
        examples[(non_word,correct_word)].append(real_words[i])
                # (real_words[i], correct_words[i]))
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
    non_word_matrix, non_word_kept = build_char_matrix(
            [non_word] * len(candidates), width=config.model_input_width)

    # Build a character matrix of the real word inputs, first marking
    # the real words with start and end of token characters.
    candidate_word_inputs = ['^'+candidate+'$' for candidate in candidates]
    candidate_word_matrix, candidate_word_kept = build_char_matrix(
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

    for non_word,correct_word in test_results.keys():
        d = {}
        d['non_word'] = non_word
        d['correct_word'] = correct_word
        d['correct_word_in_dict'] = 1

        sugg_tuples = test_results[(non_word,correct_word)]

        d['correct_word_is_in_suggestions'] = int(any([s[0] == correct_word for s in sugg_tuples]))

        for sugg_tuple in sugg_tuples:
            d['suggestion'] = sugg_tuple[0]
            d['suggestion_index'] = sugg_tuple[1]
            #d['correct_word'] = sugg_tuple[2]
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

def remove_split_and_hyphenated_suggestions(df):
    df = df.copy()

    df.suggestion_index = df.suggestion_index.astype(np.float)

    # Find the groups first, before setting the suggestion
    # index (i.e. the rank) of the space- or hyphen-containing
    # suggestions to NaN.
    idx = np.where(df.suggestion_index == 0)[0]
    offsets = [idx[i:i+2] for i in range(len(idx))]
    offsets[-1] = [offsets[-1].item(), len(df)]

    # Now set the suggestion index of the space- or hyphen-containing
    # suggestions to NaN.
    exclude_from_candidate_list = df.suggestion.apply(
            contains_space_or_hyphen)
    mask_idx = np.where(exclude_from_candidate_list)[0]
    masked_sugg_idx = df.suggestion_index.values
    masked_sugg_idx[mask_idx] = np.nan
    df.suggestion_index = masked_sugg_idx

    groups = []

    pbar = build_pbar(offsets)

    for i,(start,end) in enumerate(offsets):
        pbar.update(i+1)
        df_tmp = df.iloc[start:end, :].copy()
        excluded = df_tmp.suggestion_index.isnull()
        if any(excluded):
            n = sum(~excluded)
            sugg_idx = df_tmp.suggestion_index.values
            sugg_idx[~np.isnan(sugg_idx)] = np.arange(n)
            df_tmp.suggestion_index = sugg_idx
        groups.append(df_tmp)

    pbar.finish()

    return pd.concat(groups, axis=0)

"""
def run_mitton_corpora(retriever, model, model_cfg):
    dfs = {}
    results = {}

    for corpus in spelling.mitton.CORPORA:
        corpus_name = spelling.evaluate.corpus_name(corpus)

        non_words = []
        real_words = []
        correct_words = []
        is_real_word_error = []

        words = spelling.mitton.load_mitton_words('../../spelling/' + corpus)
        pairs = spelling.mitton.build_mitton_pairs(words)
        for non_word,correct_word in pairs:
            candidates = retriever[non_word]
            if len(candidates) > len(set(candidates)):
                raise ValueError("candidate list is not unique: %s" % (
                    ', '.join(candidates)))
            for c in candidates:
                is_real_word_error.append(non_word in candidates)
                non_words.append(non_word)
                correct_words.append(correct_word)
                real_words.append(c)

        examples = build_examples(non_words, real_words, correct_words)
        result = run_model(model, model_cfg, examples)
        results[corpus_name] = result
        df_model = format_model_suggs_for_analysis(result)
        df_model.sort_values(['non_word', 'correct_word', 'suggestion'], inplace=True)
        dfs[corpus_name] = df_model

    return results, dfs
"""


def setup(results, mode='test'):
    df_key = 'df_' + mode
    df = results[df_key]
    model = results['model']
    le = results['label_encoder']
    input_width = model.input_config[0]['input_shape'][0]
    return df, model, le, input_width

def compute_ranks(model, width, non_words, real_words, le, batch_size=100):
    ranks = []
    non_words = np.array(['^'+w+'$' for w in non_words])
    real_words = np.array(real_words)
    print('non_words', non_words.shape, len(non_words))

    batches = list(range(0, len(non_words), batch_size))
    pbar = build_pbar(batches)

    for i,batch_start in enumerate(batches):
        pbar.update(i+1)
        non_word_batch = non_words[batch_start:batch_start+batch_size]
        real_word_batch = real_words[batch_start:batch_start+batch_size]

        input_matrix = build_char_matrix(non_word_batch,
                width=width, return_mask=False)

        outputs = model.predict({'non_word_input': input_matrix })
        probs = outputs['multiclass_correction_target']

        for j,idx in enumerate(le.transform(real_word_batch)):
            non_word = non_word_batch[j]
            correct_word = real_word_batch[j]

            ranking = np.argsort(probs[j])[::-1]
            rank_of_correct_word = np.where(ranking == idx)[0].item()
            prob_of_correct_word = probs[j][idx]
            max_prob_idx = np.argmax(probs[j])
            predicted_word = le.inverse_transform(max_prob_idx)
            prob_of_predicted_word = probs[j][max_prob_idx]

            ranks.append((non_word, correct_word, predicted_word,
                rank_of_correct_word,
                prob_of_correct_word,
                prob_of_predicted_word))

    pbar.finish()

    return ranks

    """
    preds = []
    for i,j in enumerate(rank):
        if i > max_candidates:
            break
        prob = probs[j]
        if prob < min_prob:
            continue
        word = le.inverse_transform(j)
        preds.append((word,prob))

    preds = sorted(preds, key=lambda t: -t[1])

    return {
            'preds': preds,
            'rank_of_correct_word': rank_of_correct_word,
            'prob_of_correct_word': probs[correct_word]
            }
    """

def build_convnet_ranks(mode='test'):
    results = run_analysis(mode)
    df, model, le, input_width = setup(results, mode)

    non_words = df.word.tolist()
    real_words = df.real_word.tolist()

    convnet_ranks = compute_ranks(model, input_width, non_words, real_words, le)
    with open('/tmp/convnet-ranks-testset.pkl', 'wb') as f:
        pickle.dump(convnet_ranks, f, protocol=2)
    return convnet_ranks

def build_convnet_mitton_ranks(mode='test'):
    # Load the model.  This also loads the original dataset
    # used to train the model, but we ignore it since we're
    # interested in the Mitton corpora.

    results = run_analysis(mode)
    _, model, le, input_width = setup(results, mode)

    vocabulary = set(le.classes_)

    spelling_dir = os.path.dirname(os.path.dirname(spelling.__file__))
    ranks = {}
    for corpus_dat_file in spelling.mitton.CORPORA:
        corpus_name = corpus_dat_file.replace('data/', '').replace('.dat', '')
        corpus_path = spelling_dir + '/' + corpus_dat_file
        print(corpus_name, corpus_path)
        words = spelling.mitton.load_mitton_words(corpus_path)
        pairs = spelling.mitton.build_mitton_pairs(words)
        print(pairs[0])

        non_words = [p[0] for p in pairs]
        real_words = [p[1] for p in pairs]

        kept_non_words = []
        kept_real_words = []
        pbar = build_pbar(non_words)
        for i,non_word in enumerate(non_words):
            pbar.update(i+1)
            real_word = real_words[i]
            if real_word in vocabulary:
                kept_non_words.append(non_word)
                kept_real_words.append(real_word)
        pbar.finish()

        ranks[corpus_name] = compute_ranks(model,
                input_width, kept_non_words, kept_real_words, le)
        with open('/tmp/convnet-ranks-mitton-%s.pkl' % corpus_name, 'wb') as f:
            pickle.dump(ranks[corpus_name], f, protocol=2)

    return ranks

def build_retriever_ranks(mode='test'):
    results = run_analysis(mode)
    df, model, le, input_width = setup(results, mode)

    non_words = df.word.tolist()
    real_words = df.real_word.tolist()

    retriever = results['retriever']
    pbar = build_pbar(non_words)
    retriever_ranks = []
    for i,non_word in enumerate(non_words):
        pbar.update(i+1)
        correct_word = real_words[i]
        candidates = retriever[non_word]
    
        try:
            correct_word_rank = candidates.index(correct_word)
        except ValueError:
            correct_word_rank = -1
    
        retriever_ranks.append((
            non_word, correct_word, candidates[0], correct_word_rank))

    with open('/tmp/retriever-ranks-testset.pkl', 'wb') as f:
        pickle.dump(retriever_ranks, f, protocol=2)

    return retriever_ranks

def build_retriever_mitton_ranks(mode='test'):
    # Load the model.  This also loads the original dataset
    # used to train the model, but we ignore it since we're
    # interested in the Mitton corpora.

    results = run_analysis(mode)
    _, model, le, input_width = setup(results, mode)
    retriever = results['retriever']

    vocabulary = set(le.classes_)

    spelling_dir = os.path.dirname(os.path.dirname(spelling.__file__))
    ranks = {}
    for corpus_dat_file in spelling.mitton.CORPORA:
        corpus_name = corpus_dat_file.replace('data/', '').replace('.dat', '')
        corpus_path = spelling_dir + '/' + corpus_dat_file
        print(corpus_name, corpus_path)
        words = spelling.mitton.load_mitton_words(corpus_path)
        pairs = spelling.mitton.build_mitton_pairs(words)
        print(pairs[0])

        non_words = [p[0] for p in pairs]
        real_words = [p[1] for p in pairs]

        kept_non_words = []
        kept_real_words = []
        pbar = build_pbar(non_words)
        for i,non_word in enumerate(non_words):
            pbar.update(i+1)
            real_word = real_words[i]
            if real_word in vocabulary:
                kept_non_words.append(non_word)
                kept_real_words.append(real_word)
        pbar.finish()

        pbar = build_pbar(kept_non_words)
        retriever_ranks = []
        for i,non_word in enumerate(kept_non_words):
            pbar.update(i+1)
            correct_word = kept_real_words[i]
            candidates = retriever[non_word]
    
            try:
                correct_word_rank = candidates.index(correct_word)
            except ValueError:
                correct_word_rank = -1

            retriever_ranks.append((
                non_word, correct_word, candidates[0], correct_word_rank))

        with open('/tmp/retriever-ranks-%s.pkl' % corpus_name, 'wb') as f:
            pickle.dump(retriever_ranks, f, protocol=2)

        ranks[corpus_name] = retriever_ranks

    return ranks

def build_data_frames(convnet_ranks, retriever_ranks):
    # The non-words we use are between 5 and 25 characters
    # long, but they are marked with an initial '^' and a 
    # terminating '$' for the ConvNet.  So when we compute
    # accuracy-at-k for the ConvNet, we have to add 2 to the
    # length.  We also use the length -1 to indicate that
    # the accuracy is computed over non-words of all lengths.
    non_word_lengths = [-1] + [length for length in range(5,26)]
    print('non_word_lengths', non_word_lengths)
    
    algorithms = []
    lengths = []
    ks = []
    accuracies = []
    
    ranks = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
    
    # Compute accuracy at K for the ConvNet model.
    for length in non_word_lengths:
        if length == -1:
            all_at_length = convnet_ranks
        else:
            # Add 2 to account for the '^' and '$'.
            all_at_length = [item for item in convnet_ranks
                    if len(item[0]) == length+2]
    
        for k in ranks:
            total_at_k = len([item for item in all_at_length if item[3] < k])

            algorithms.append("ConvNet")
            lengths.append(length)
            ks.append(k)

            try:
                accuracy = total_at_k/float(len(all_at_length))
            except ZeroDivisionError:
                accuracy = 0.0
            accuracies.append(accuracy)
    
    # Compute accuracy at K for the retriever model.
    for length in non_word_lengths:
        if length == -1:
            all_at_length = retriever_ranks
        else:
            all_at_length = [item for item in retriever_ranks
                    if len(item[0]) == length]

        for k in ranks:
            # The correct word doesn't necessarily appear in the
            # list of candidates returned by a retriever.  The
            # absence of the correct word is indicated by a rank
            # of -1.  (This is done in the try/except block above.)
            total_at_k = len([item for item in all_at_length
                if item[-1] < k and item[-1] != -1])

            algorithms.append("Aspell with Jaro-Winkler")
            lengths.append(length)
            ks.append(k)

            try:
                accuracy = total_at_k/float(len(all_at_length))
            except ZeroDivisionError:
                accuracy = 0.0
            accuracies.append(accuracy)

    df = pd.DataFrame(data={
        'Algorithm': algorithms,
        'Rank': ks,
        'Accuracy': accuracies,
        'Non-word Length': lengths
        })

    df['Corpus'] = 'Generated'

    return df
