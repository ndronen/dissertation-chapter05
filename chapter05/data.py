import collections
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import spelling.dictionary as spelldict
from spelling.utils import build_progressbar as build_pbar

def load_df(csv_path, errors_only=True):
    df = pd.read_csv(csv_path, sep='\t', encoding='utf8')
    if errors_only:
        df = df[df.binary_target == 0]
    return df

def fit_label_encoder(df):
    le = LabelEncoder()
    le.fit(df.real_word)
    return le

def split_df(df, min_frequency=0, min_length=0, max_length=100, seed=17):
    frequencies = df.multiclass_correction_target.value_counts().to_dict()
    df['frequency'] = df.multiclass_correction_target.apply(lambda x: frequencies[x])
    df['len'] = df.word.apply(len)
    mask = (df.frequency >= min_frequency) & (df.len >= min_length) & (df.len <= max_length)
    df = df[mask]
    print(len(df), len(df.multiclass_correction_target.unique()))

    df_train, df_other = train_test_split(df, train_size=0.8, random_state=seed)
    df_valid, df_test = train_test_split(df_other, train_size=0.5, random_state=seed)

    return df_train, df_valid, df_test

def run(csv_path):
    df = load_df(csv_path)
    train_df, valid_df, test_df = split_df(df)
    return df, train_df, valid_df, test_df

def compute_ranks_of_true_correction(df):
    # Retrievers.
    aspell = spelldict.AspellRetriever()
    vocabulary = df.real_word.unique().tolist()
    edit_distance = spelldict.EditDistanceRetriever(vocabulary,
            stop_retrieving_when_found=False)
    retrievers = { 'aspell': aspell, 'edit_distance': edit_distance }

    # Sorters.
    jaro_winkler_sorter = spelldict.DistanceSorter('jaro_winkler')
    sorters = { 'jaro_winkler': jaro_winkler_sorter }

    ranks = collections.defaultdict(list)

    non_words = df.word.tolist()
    corrections = df.real_word.tolist()

    pbar = build_pbar(non_words)

    def index_or_none(candidates, correction):
        try:
            return candidates.index(correction)
        except ValueError:
            return None

    for i,non_word in enumerate(non_words):
        pbar.update(i+1)
        correction = corrections[i]
        # Retrievers alone and with sorters.
        for retvr in retrievers.keys():
            candidates = retrievers[retvr][non_word]
            true_correction_index = index_or_none(candidates, correction)
            ranks[retvr].append(true_correction_index)
            for srtr in sorters.keys():
                reranked = sorters[srtr].sort(non_word, candidates)
                true_correction_index = index_or_none(reranked, correction)
                name = '_'.join([retvr, srtr])
                ranks[name].append(true_correction_index)
    pbar.finish()

    return pd.DataFrame(data=ranks)

def build_candidate_dataset(non_word, correction, retriever):
    def add_example(dataset, non_word, correction, target, rank):
        dataset['word'].append(non_word)
        dataset['real_word'].append(correction)
        dataset['binary_correction_target'].append(target)
        dataset['rank'].append(rank)

    candidates = retriever[non_word]
    dataset = collections.defaultdict(list)

    for rank,candidate in enumerate(candidates):
        target = 1 if candidate == correction else 0
        add_example(dataset, non_word, candidate, target, rank)
    if correction not in candidates:
        add_example(dataset, non_word, correction, 1, None)

    df = pd.DataFrame(data=dataset)


    if real_word in candidates:
        correct_word_is_in_suggestions = 1
    else:
        correct_word_is_in_suggestions = 0

    real_word_candidates = retriever[real_word]
    if real_word in real_word_candidates:
        correct_word_in_dict = 1
    else:
        correct_word_in_dict = 0

    df['correct_word_is_in_suggestions'] = correct_word_is_in_suggestions
    df['correct_word_in_dict'] = correct_word_in_dict

    return df
