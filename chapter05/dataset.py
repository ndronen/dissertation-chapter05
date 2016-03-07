import string
import sys
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from modeling.utils import balanced_class_weights
from keras.utils import np_utils
import spelling.preprocess
import spelling.features
import spelling.dictionary
import traceback

class DatasetGenerator(object):
    pass

class MulticlassModelDatasetGenerator(DatasetGenerator):
    """
    non_words : list of unicode
        Non-word spelling errors.
    corrections : list of unicode
        The corrections for the non-words.
    targets : list of int
        The target variables of the corrections.
    model_input_width : int
        The width of the character matrix in which the non-words and
        real words are embedded.
    n_classes : int
        The number of classes in the data set.
    noise_word_target : int
        The target variable of out-of-vocabulary (OOV) words.  When this
        parameter is greater than -1, a random noise training example
        is generated for each real training example.  The length of the
        training example is the same as the real training example.
        The edit distance from a noise word to the nearest word in the
        dictionary must be at least 2 for a word up to four characters
        and half of the characters for longer words.
    retriever : dict
        A mapping from non-words to possible replacements.  See the
        retriever implementations in spelling.dictionary.  The retriever
        is used to find words in the dictionary that are close to noise
        words, which in turn ensures that the noise words are not too 
        similar to known words.
    random_state : int or numpy.random.RandomState
        The state of the random number generator.
    """
    def __init__(self, non_words, corrections, targets, model_input_width, n_classes, noise_word_target=-1, retriever=None, batch_size=128, random_state=1, use_correct_word_as_non_word_example=False):
        self.__dict__.update(locals())
        del self.self

        assert len(non_words) > 0
        assert len(non_words) == len(corrections)
        assert noise_word_target < 0 or retriever is not None

        self.non_words = np.array(non_words)
        self.corrections = np.array(corrections)
        self.targets = np.array(targets)

        self.random_state = check_random_state(random_state)
        self.generate_noise = self.noise_word_target > -1 

    def generate(self, exhaustive=False, train=False):
        if exhaustive:
            for i in range(len(self.non_words)):
                try:
                    n = self.generate_next(i, train)
                    if n is not None:
                        yield n
                except Exception as e:
                    print(e, type(e))
                    (t, val, tb) = sys.exc_info()
                    traceback.print_tb(tb)
                    raise e
        else:
            while 1:
                try:
                    i = self.random_state.choice(len(self.non_words), size=self.batch_size)
                    n = self.generate_next(i, train)
                    if n is not None:
                        yield n
                except Exception as e:
                    print(e, type(e))
                    (t, val, tb) = sys.exc_info()
                    traceback.print_tb(tb)
                    raise e

    def generate_next(self, i, train=False):
        # This non-word is a spelling error that corrects to a known word.
        non_words = self.non_words[i].tolist()
        corrections = self.corrections[i].tolist()
        targets = self.targets[i].tolist()

        if isinstance(non_words, str):
            non_words = [non_words]
        if isinstance(corrections, str):
            corrections = [corrections]
        if isinstance(targets, int):
            targets = [targets]

        non_word_inputs = ['^'+nw+'$' for nw in non_words]
        non_word_matrix, non_word_kept = spelling.preprocess.build_char_matrix(
                non_word_inputs, width=self.model_input_width)

        if sum(non_word_kept) == 0:
            return None

        multiclass_correction_targets = np_utils.to_categorical(
                targets, self.n_classes)

        """
        if train is False:
            print('non_word', np.array(non_words)[non_word_kept].shape)
            print('non_word_input', non_word_matrix[non_word_kept].shape)
            print('correct_word', np.array(corrections)[non_word_kept].shape)
            print('multiclass_correction_target', multiclass_correction_targets[non_word_kept].shape)
        """

        return {
                'non_word': np.array(non_words)[non_word_kept],
                'non_word_input': non_word_matrix[non_word_kept],
                'correct_word': np.array(corrections)[non_word_kept],
                'multiclass_correction_target': multiclass_correction_targets[non_word_kept]
                }

    def generate_noise_words(self, non_words):
        # A noise word is far enough from any known word that it
        # should only correct the "unknown word" target.  The rule
        # is that the edit distance must be at least three for a
        # word up to four characters and at least half the characters 
        # for any longer word.
        noise_non_words = []
        noise_corrections = []
        noise_targets = []

        def use_this_noise_word(noise_word):
            noise_non_words.append(noise_word)
            noise_corrections.append("UNKNOWNWORD")
            noise_targets.append(self.noise_word_target)

        for i,non_word in enumerate(non_words.copy()):
            if len(non_word) == 1:
                continue

            while True:
                noise_word_chars = self.random_state.choice(
                        list(string.ascii_lowercase),
                        size=len(non_word))
                noise_word = ''.join(noise_word_chars)
                candidates = self.retriever[noise_word]
                closest = [spelling.features.distance(
                    c, noise_word, 'levenshtein_distance') for c in candidates]
                if len(candidates) == 0:
                    use_this_noise_word(noise_word)
                    break
                elif len(candidates) == 1 and noise_word in candidates:
                    if isinstance(self.retriever, spelling.dictionary.EditDistanceRetriever):
                        use_this_noise_word(noise_word)
                        break
                    else:
                        continue
                elif noise_word in candidates:
                    continue
                else:
                    if len(non_word) <= 4:
                        min_distance = 2
                    else:
                        min_distance = int(len(non_word)/2.)
                    if min(closest) > min_distance:
                        use_this_noise_word(noise_word)
                        break

        return noise_non_words, noise_corrections, noise_targets

class Pool(object):
    def __init__(self, non_words, corrections, targets):
        assert len(non_words) > 0
        assert len(non_words) == len(corrections)
        self._non_words = np.array(non_words)
        self._corrections = np.array(corrections)
        self._targets = np.array(targets)

    @property
    def non_words(self):
        raise NotImplementedError()

    @non_words.setter
    def non_words(self, value):
        raise NotImplementedError()

    @property
    def corrections(self):
        raise NotImplementedError()

    @corrections.setter
    def corrections(self, value):
        raise NotImplementedError()

    @property
    def targets(self):
        raise NotImplementedError()

    @targets.setter
    def targets(self, value):
        raise NotImplementedError()

    def __len__(self):
        return len(self._non_words)

class LengthIndexedPool(Pool):
    def __init__(self, non_words, corrections, targets, min_length=0, max_length=sys.maxsize):
        assert min_length >= 0
        assert max_length >= 1
        super(LengthIndexedPool, self).__init__(non_words, corrections, targets)
        self.lengths = np.array([len(x) for x in self._non_words])
        self._min_length = min_length
        self._max_length = max_length
        self.update_index()

    def update_index(self):
        self.mask = (self.lengths >= self._min_length) & (self.lengths <= self._max_length)
        self.__non_words = self._non_words[self.mask]
        self.__corrections = self._corrections[self.mask]
        self.__targets = self._targets[self.mask]

    @property
    def min_length(self):
        return self._min_length

    @min_length.setter
    def min_length(self, value):
        if value > self._max_length:
            raise ValueError("min_length (%d) must be > max_length (%d)" % (
                value, self._max_length))
        self._min_length = max(value, 0)

        self.update_index()

    @property
    def max_length(self):
        return self._max_length

    @max_length.setter
    def max_length(self, value):
        if value < self._min_length:
            raise ValueError("max_length (%d) must be < min_length (%d)" % (
                value, self._min_length))
        self._max_length = max(value, 0)
        self.update_index()

    @property
    def non_words(self):
        return self.__non_words

    @property
    def corrections(self):
        return self.__corrections

    @property
    def targets(self):
        return self.__targets

    def __len__(self):
        return len(self.__non_words)

class ScheduledMulticlassModelDatasetGenerator(DatasetGenerator):
    """
    pool : chapter05.dataset.Pool
        A Pool instance that provides non_words, correction words,
        and the target variable of the correction words.
    model_input_width : int
        The width of the character matrix in which the non-words and
        real words are embedded.
    n_classes : int
        The number of classes in the data set.
    random_state : int or numpy.random.RandomState
        The state of the random number generator.
    """
    def __init__(self, pool, model_input_width, n_classes, batch_size=128, random_state=1):
        self.__dict__.update(locals())
        del self.self
        assert pool is not None
        assert model_input_width >= 1
        assert n_classes > 0
        assert batch_size > 0
        self.random_state = check_random_state(random_state)

    def generate(self, exhaustive=False, train=False):
        if exhaustive:
            for i in range(len(self.pool)):
                try:
                    n = self.generate_next(i, train)
                    if n is not None:
                        yield n
                except Exception as e:
                    print(e, type(e))
                    (t, val, tb) = sys.exc_info()
                    traceback.print_tb(tb)
                    raise e
        else:
            while 1:
                try:
                    i = self.random_state.choice(len(self.pool), size=self.batch_size)
                    n = self.generate_next(i, train)
                    if n is not None:
                        yield n
                except Exception as e:
                    print(e, type(e))
                    (t, val, tb) = sys.exc_info()
                    traceback.print_tb(tb)
                    raise e

    def generate_next(self, i, train=False):
        # This non-word is a spelling error that corrects to a known word.
        non_words = self.pool.non_words[i].tolist()
        corrections = self.pool.corrections[i].tolist()
        targets = self.pool.targets[i].tolist()

        if isinstance(non_words, str):
            non_words = [non_words]
        if isinstance(corrections, str):
            corrections = [corrections]
        if isinstance(targets, int):
            targets = [targets]

        non_word_inputs = ['^'+nw+'$' for nw in non_words]
        non_word_matrix, non_word_kept = spelling.preprocess.build_char_matrix(
                non_word_inputs, width=self.model_input_width)

        if sum(non_word_kept) == 0:
            return None

        multiclass_correction_targets = np_utils.to_categorical(
                targets, self.n_classes)

        """
        if train is False:
            print('non_word', np.array(non_words)[non_word_kept].shape)
            print('non_word_input', non_word_matrix[non_word_kept].shape)
            print('correct_word', np.array(corrections)[non_word_kept].shape)
            print('multiclass_correction_target', multiclass_correction_targets[non_word_kept].shape)
        """

        return {
                'non_word': np.array(non_words)[non_word_kept],
                'non_word_input': non_word_matrix[non_word_kept],
                'correct_word': np.array(corrections)[non_word_kept],
                'multiclass_correction_target': multiclass_correction_targets[non_word_kept]
                }

class BinaryModelDatasetGenerator(DatasetGenerator):
    """
    non_words : list of unicode
        Non-word spelling errors.
    corrections : list of unicode
        The corrections for the non-words.
    retriever : dict
        A mapping from non-words to possible replacements.  See the
        retriever implementations in spelling.dictionary.
    model_input_width : int
        The width of the character matrix in which the non-words and
        real words are embedded.
    sample_weight_exponent : int
        The power to which to raise the sample weights.
    random_state : int or numpy.random.RandomState
        The state of the random number generator.
    use_correct_word_as_non_word_example : bool
        Whether to include the correct word as an example in the training set.
        These examples train the model to learn to correct a real word to a
        real word and may function as an optimizer or a regularizer.
    distance_targets: list of str
        The names of distances in spelling.features.  If provided, then
        for each name in this list, the distance between a non-word and
        a proposed correction is included as a target variable.
    """
    def __init__(self, non_words, corrections, retriever, model_input_width, sample_weight_exponent=1, random_state=1, use_correct_word_as_non_word_example=False, distance_targets=[], max_candidates=sys.maxsize):
        self.__dict__.update(locals())
        del self.self
        self.random_state = check_random_state(random_state)
        assert len(non_words) == len(corrections)

        assert distance_targets is not None
        assert isinstance(distance_targets, (list, tuple))

    def generate(self, exhaustive=False, train=False):
        if exhaustive:
            for i in range(len(self.non_words)):
                try:
                    n = self.generate_next(i, train)
                    if n is not None:
                        yield n
                except Exception as e:
                    print(e, type(e))
                    (t, val, tb) = sys.exc_info()
                    traceback.print_tb(tb)
                    raise e
        else:
            while 1:
                try:
                    i = self.random_state.choice(len(self.non_words), size=1)
                    n = self.generate_next(i, train)
                    if n is not None:
                        yield n
                except Exception as e:
                    print(e, type(e))
                    (t, val, tb) = sys.exc_info()
                    traceback.print_tb(tb)
                    raise e

    def generate_next(self, i, train=False):
        non_word = self.non_words[i]
        correction = self.corrections[i]
        candidates = self.retriever[non_word]

        if self.use_correct_word_as_non_word_example:
            # Add the correct word to the list of candidates, even if it's there
            # already, so we can teach the model that the true correction for a
            # known word is the word itself.
            candidates.append(correction)

        if len(candidates) <= 1:
            return None

        candidates = candidates[0:self.max_candidates]

        targets = np.zeros((len(candidates),2), dtype=np.int64)
        for j, candidate in enumerate(candidates):
            if candidate == correction:
                targets[j, 1] = 1
            else:
                targets[j, 0] = 1

        class_weights = balanced_class_weights(targets[:, 1], n_classes=2,
                class_weight_exponent=self.sample_weight_exponent)
        sample_weights = np.zeros(len(candidates))
        for k, candidate in enumerate(candidates):
            sample_weights[k] = class_weights[targets[k, 1]]

        # Build a character matrix of the non-word inputs, first marking
        # the non-words with start and end of token characters.
        non_word_inputs = ['^'+non_word+'$' for candidate in candidates]
        if self.use_correct_word_as_non_word_example:
            non_word_inputs.append('^'+correction+'$')
        non_word_matrix, non_word_kept = spelling.preprocess.build_char_matrix(
                [non_word] * len(candidates), width=self.model_input_width)

        # Build a character matrix of the real word inputs, first marking
        # the real words with start and end of token characters.
        candidate_word_inputs = ['^'+candidate+'$' for candidate in candidates]
        candidate_word_matrix, candidate_word_kept = spelling.preprocess.build_char_matrix(
                candidates, width=self.model_input_width)

        mask = non_word_kept & candidate_word_kept
        idx = np.where(non_word_kept & candidate_word_kept)[0]

        non_word_matrix = non_word_matrix[mask]
        candidate_word_matrix = candidate_word_matrix[mask]
        targets = targets[mask]
        sample_weights = sample_weights[mask]

        _non_word = np.array([non_word] * len(idx))
        _correct_word = np.array([correction] * len(idx))
        _candidate_word = np.array(candidates)[mask]

        lengths = [len(x) for x in [non_word_matrix, candidate_word_matrix, targets, sample_weights, _non_word, _correct_word, _candidate_word]]
        if not all(l == lengths[0] for l in lengths):
            print('non_word %s non_word_matrix %d candidate_word_matrix %d targets %d sample_weights %d _non_word %d _correct_word %d _candidate_word %d' % 
                    (non_word,
                    len(non_word_matrix),
                    len(candidate_word_matrix),
                    len(targets),
                    len(sample_weights),
                    len(_non_word),
                    len(_correct_word),
                    len(_candidate_word)))

        assert \
                len(non_word_matrix) == \
                len(candidate_word_matrix) == \
                len(targets) == \
                len(sample_weights) == \
                len(_non_word) == \
                len(_correct_word) == \
                len(_candidate_word)

        def log_normalize(x):
            return 1 + np.log(1 + x)


        data_dict = {
                'non_word': np.array([non_word] * len(idx)),
                'correct_word': np.array([correction] * len(idx)),
                'candidate_word': np.array(candidates)[mask],
                'non_word_input': non_word_matrix[mask],
                'candidate_word_input': candidate_word_matrix[mask],
                'binary_correction_target': targets[mask],
                'candidate_rank': log_normalize(np.arange(len(candidates))[mask])
                }

        sample_weight_dict = {
                'binary_correction_target': sample_weights[mask],
                'candidate_rank': sample_weights[mask]
                }

        non_words = None
        candidate_words = None

        for name in self.distance_targets:
            if non_words is None:
                non_words = data_dict['non_word'].tolist()
                candidate_words = data_dict['candidate_word'].tolist()

            target_values = []
            for i,non_word in enumerate(non_words):
                candidate_word = candidate_words[i]
                d = spelling.features.distance(
                        non_word, candidate_word, name)
                target_values.append(d)
            data_dict[name] = log_normalize(np.array(target_values))

            # We want the model to learn to predict the edit distance
            # equally well for all examples.
            sample_weight_dict[name] = np.ones(len(idx))

        #print('data_dict', list(data_dict.keys()),
        #        'sample_weight_dict', list(sample_weight_dict.keys()))
        return (data_dict, sample_weight_dict)

class BinaryModelRealWordDatasetGenerator(DatasetGenerator):
    """
    vocabulary : list of unicode
        A list of real words.
    model_input_width : int
        The width of the character matrix in which the non-words and
        real words are embedded.
    random_state : int or numpy.random.RandomState
        The state of the random number generator.
    distance_targets: list of str
        The names of distances in spelling.features.  If provided, then
        for each name in this list, the distance between a non-word and
        a proposed correction is included as a target variable.
    """
    def __init__(self, vocabulary, model_input_width, random_state=1, distance_targets=[]):
        assert len(vocabulary) > 0
        assert distance_targets is not None
        assert isinstance(distance_targets, (list, tuple))

        self.__dict__.update(locals())
        del self.self

        self.random_state = check_random_state(random_state)

    def generate(self, exhaustive=False, train=False):
        if exhaustive:
            for i in range(len(self.vocabulary)):
                try:
                    n = self.generate_next(i, train)
                    if n is not None:
                        yield n
                except Exception as e:
                    print(e, type(e))
                    (t, val, tb) = sys.exc_info()
                    traceback.print_tb(tb)
                    raise e
        else:
            while 1:
                try:
                    i = self.random_state.choice(len(self.vocabulary), size=1)
                    n = self.generate_next(i, train)
                    if n is not None:
                        yield n
                except Exception as e:
                    print(e, type(e))
                    (t, val, tb) = sys.exc_info()
                    traceback.print_tb(tb)
                    raise e

    def generate_next(self, i, train=False):
        if len(i) == 1:
            batch_words = [self.vocabulary[i]]
        else:
            batch_words = self.vocabulary[i]

        batch_inputs = ['^'+w+'$' for w in batch_words]
        batch_matrix, batch_kept = spelling.preprocess.build_char_matrix(
            batch_inputs, width=self.model_input_width)

        batch_words = np.array(batch_words)[batch_kept]
        batch_matrix = batch_matrix[batch_kept]
        targets = np.ones((sum(batch_kept),2), dtype=np.int64)

        data_dict = {
                'non_word': batch_words,
                'correct_word': batch_words,
                'candidate_word': batch_words,
                'non_word_input': batch_matrix,
                'candidate_word_input': batch_matrix,
                'binary_correction_target': targets,
                'candidate_rank': 1 + np.zeros(sum(batch_kept))
                }

        sample_weight_dict = {
            'binary_correction_target': np.ones(sum(batch_kept)),
            'candidate_rank': np.ones(sum(batch_kept))
            }

        for name in self.distance_targets:
            data_dict[name] = np.zeros(sum(batch_kept))
            sample_weight_dict[name] = np.ones(sum(batch_kept))

        return (data_dict, sample_weight_dict)
