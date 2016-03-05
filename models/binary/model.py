from chapter05.dataset import BinaryModelDatasetGenerator

import sys
sys.setrecursionlimit(5000)

import numpy as np
import pandas as pd
import sklearn.neighbors
from sklearn.cross_validation import train_test_split

from keras.models import Sequential, Graph
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
import keras.callbacks
from spelling.utils import build_progressbar

import modeling.data
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights,
    build_hierarchical_softmax_layer)
from modeling.callbacks import DenseWeightNormCallback

from spelling.dictionary import (
        HashBucketRetriever,
        AspellRetriever,
        EditDistanceRetriever,
        NearestNeighborsRetriever,
        RetrieverCollection)
import spelling.features

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

def add_bn_relu(graph, config, prev_layer):
    bn_name = prev_layer + '_bn'
    relu_name = prev_layer + '_relu'
    if config.batch_normalization:
        graph.add_node(BatchNormalization(), name=bn_name, input=prev_layer)
        prev_layer = bn_name
    graph.add_node(Activation('relu'), name=relu_name, input=prev_layer)
    return relu_name

def build_model(config):
    np.random.seed(config.seed)

    graph = Graph()

    graph.add_input(config.non_word_input_name,
            input_shape=(config.model_input_width,), dtype='int')
    graph.add_input(config.candidate_word_input_name,
            input_shape=(config.model_input_width,), dtype='int')

    graph.add_shared_node(
        build_embedding_layer(config, input_width=config.model_input_width),
        name='embedding',
        inputs=[config.non_word_input_name, config.candidate_word_input_name],
        outputs=['non_word_embedding', 'candidate_word_embedding'])

    graph.add_shared_node(
            build_convolutional_layer(config),
            name='conv',
            inputs=['non_word_embedding', 'candidate_word_embedding'],
            outputs=['non_word_conv', 'candidate_word_conv'])

    non_word_prev_layer = add_bn_relu(graph, config, 'non_word_conv')
    graph.add_node(build_pooling_layer(config, input_width=config.model_input_width),
            name='non_word_pool', input=non_word_prev_layer)
    graph.add_node(Flatten(), name='non_word_flatten', input='non_word_pool')
    # Add noise only to non-words.
    graph.add_node(GaussianNoise(0.05), name='non_word_noise', input='non_word_flatten')

    candidate_word_prev_layer = add_bn_relu(graph, config, 'candidate_word_conv')
    graph.add_node(build_pooling_layer(config, input_width=config.model_input_width),
            name='candidate_word_pool', input=candidate_word_prev_layer)
    graph.add_node(Flatten(), name='candidate_word_flatten', input='candidate_word_pool')

    # Add some number of fully-connected layers without skip connections.
    prev_layer = None

    for i,n_hidden in enumerate(config.fully_connected):
        layer_name = 'dense%02d' % i
        l = build_dense_layer(config, n_hidden=n_hidden)
        if i == 0:
            graph.add_node(l, name=layer_name,
                inputs=['non_word_noise', 'candidate_word_flatten'])
        else:
            graph.add_node(l, name=layer_name, input=prev_layer)
        prev_layer = layer_name
        if config.batch_normalization:
            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
            prev_layer = layer_name+'bn'
        if config.dropout_fc_p > 0.:
            graph.add_node(Dropout(0.5), name=layer_name+'do', input=prev_layer)
            prev_layer = layer_name+'do'
    
    # Add sequence of residual blocks.
    for i in range(config.n_residual_blocks):
        # Add a fixed number of layers per residual block.
        block_name = '%02d' % i

        graph.add_node(Identity(), name=block_name+'input', input=prev_layer)
        prev_layer = block_input_layer = block_name+'input'

        try:
            n_layers_per_residual_block = config.n_layers_per_residual_block
        except AttributeError:
            n_layers_per_residual_block = 2

        for layer_num in range(n_layers_per_residual_block):
            layer_name = 'h%s%02d' % (block_name, layer_num)
    
            l = build_dense_layer(config, n_hidden=config.n_hidden_residual)
            graph.add_node(l, name=layer_name, input=prev_layer)
            prev_layer = layer_name
    
            if config.batch_normalization:
                graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
                prev_layer = layer_name+'bn'
    
            if i < n_layers_per_residual_block:
                a = Activation('relu')
                graph.add_node(Activation('relu'), name=layer_name+'relu', input=prev_layer)
                prev_layer = layer_name+'relu'
                if config.dropout_fc_p > 0.:
                    graph.add_node(Dropout(config.dropout_fc_p), name=layer_name+'do', input=prev_layer)
                    prev_layer = layer_name+'do'

        graph.add_node(Identity(), name=block_name+'output', inputs=[block_input_layer, prev_layer], merge_mode='sum')
        graph.add_node(Activation('relu'), name=block_name+'relu', input=block_name+'output')
        prev_layer = block_input_layer = block_name+'relu'

    graph.add_node(Dense(2), name='softmax', input=prev_layer)
    prev_layer = 'softmax'
    if config.batch_normalization:
        graph.add_node(BatchNormalization(), 
                name='softmax_bn', input='softmax')
        prev_layer = 'softmax_bn'
    graph.add_node(Activation('softmax'),
            name='softmax_activation', input=prev_layer)

    graph.add_output(name=config.target_name, input='softmax_activation')

    load_weights(config, graph)

    optimizer = build_optimizer(config)

    graph.compile(loss={config.target_name: config.loss}, optimizer=optimizer)

    return graph

class FakeGenerator(object):
    def __init__(self, config):
        self.config = config

    def generate(self, n_samples=2):
        while 1:
            x = np.random.randint(self.config.n_embeddings,
                    size=(n_samples, self.config.model_input_width))
            y = np.random.binomial(1, p=0.1, size=n_samples)
            y_one_hot = np.zeros((n_samples, 2))
            for i in range(n_samples):
                y_one_hot[i, y[i]] = 1

            d = {
                self.config.non_word_input_name: x,
                self.config.candidate_word_input_name: x,
                self.config.target_name: y_one_hot
                }
            sample_weights = { "binary_correction_target": np.array([0.4, 10.]) }

            yield (d, sample_weights)

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, config, generator, n_samples):
        self.__dict__.update(locals())
        del self.self

    def on_epoch_end(self, epoch, logs={}):
        correct = []
        y = []
        y_hat = []
        y_hat_dictionary = []
        counter = 0
        pbar = build_progressbar(self.n_samples)
        print('\n')
        g = self.generator.generate(exhaustive=True)
        while True:
            pbar.update(counter)
            # Each call to next results in a batch of possible
            # corrections, only one of which is correct.
            try:
                next_batch = next(g)
            except StopIteration:
                break

            if isinstance(next_batch, (tuple, list)):
                d, sample_weight = next_batch
            else:
                assert isinstance(next_batch, dict)
                d = next_batch
                sample_weight = None

            targets = d[self.config.target_name]
            pred = self.model.predict(d, verbose=0)[self.config.target_name]

            y.extend(targets[:, 1].tolist())

            y_hat_tmp = [0] * len(targets)
            y_hat_tmp[np.argmax(pred[:, 1])] = 1
            y_hat.extend(y_hat_tmp)

            correct_word = d['correct_word'][0]

            y_hat_dictionary_tmp = []
            for i,c in enumerate(d['candidate_word']):
                # The first word in the results returned by the dictionary
                # is the dictionary's highest-scoring candidate for
                # replacing the non-word.
                if i == 0:
                    y_hat_dictionary_tmp.append(1)
                else:
                    y_hat_dictionary_tmp.append(0)
            y_hat_dictionary.extend(y_hat_dictionary_tmp)

            if len(y_hat_dictionary_tmp) != len(targets):
                raise ValueError('non_word %s correct_word %s dictlen %d targetslen %d' %
                        (d['non_word'][0], d['correct_word'][0],
                            len(y_hat_dictionary_tmp),
                            len(targets)))

            counter += 1
            if counter >= self.n_samples:
                break

        pbar.finish()

        self.config.logger('\n')
        self.config.logger('Dictionary accuracy %.04f F1 %0.4f' % 
                (accuracy_score(y, y_hat_dictionary), f1_score(y, y_hat_dictionary)))
        self.config.logger('Dictionary confusion matrix')
        self.config.logger(confusion_matrix(y, y_hat_dictionary))

        self.config.logger('\n')
        self.config.logger('ConvNet accuracy %.04f F1 %0.4f' % 
                (accuracy_score(y, y_hat), f1_score(y, y_hat)))
        self.config.logger('ConvNet confusion matrix')
        self.config.logger(confusion_matrix(y, y_hat))

        self.config.logger('\n')

def build_callbacks(config, generator, n_samples):
    callbacks = []
    callbacks.append(MetricsCallback(config, generator, n_samples))
    callbacks.append(DenseWeightNormCallback(config))
    return callbacks

def fit(config):
    df = pd.read_csv(config.non_word_csv, sep='\t', encoding='utf8')
    df = df[df.binary_target == 0]

    # Select examples by frequency and length.
    frequencies = df.multiclass_correction_target.value_counts().to_dict()
    df['frequency'] = df.multiclass_correction_target.apply(lambda x: frequencies[x])
    df['len'] = df.word.apply(len)
    mask = (df.frequency >= config.min_frequency) & (df.len >= config.min_length)
    df = df[mask]
    print(len(df), len(df.multiclass_correction_target.unique()))

    # Retrivers
    aspell_retriever = AspellRetriever()
    #edit_distance_retriever = EditDistanceRetriever(df.real_word.unique())
    #hash_bucket_retriever = HashBucketRetriever(df.real_word.unique(), spelling.features.metaphone)
    #estimator = sklearn.neighbors.NearestNeighbors(n_neighbors=10, metric='hamming', algorithm='auto')
    #nearest_neighbors_retriever = NearestNeighborsRetriever(df.real_word.unique(), estimator)
    retriever = RetrieverCollection(retrievers=[
            aspell_retriever
            #, hash_bucket_retriever,
            #edit_distance_retriever, nearest_neighbors_retriever
            ])

    df_train, df_other = train_test_split(df, train_size=0.8, random_state=config.seed)
    train_words = set(df_train.word)
    other_words = set(df_other.word)
    leaked_words = train_words.intersection(other_words)
    df_other = df_other[~df_other.word.isin(leaked_words)]
    df_valid, df_test = train_test_split(df_other, train_size=0.5, random_state=config.seed)

    print('train %d validation %d test %d' % (len(df_train), len(df_valid), len(df_test)))
    
    train_generator = BinaryModelDatasetGenerator(
            df_train.word.tolist(),
            df_train.real_word.tolist(),
            retriever,
            config.model_input_width,
            config.sample_weight_exponent)

    validation_generator = BinaryModelDatasetGenerator(
            df_valid.head(config.n_val_samples).word.tolist(),
            df_valid.head(config.n_val_samples).real_word.tolist(),
            retriever,
            config.model_input_width,
            config.sample_weight_exponent)

    graph = build_model(config)

    config.logger('model has %d parameters' % graph.count_params())

    callbacks = build_callbacks(config, validation_generator,
            n_samples=config.n_val_samples)

    graph.fit_generator(train_generator.generate(),
            samples_per_epoch=config.samples_per_epoch,
            nb_epoch=config.n_epoch,
            nb_worker=config.n_worker,
            callbacks=callbacks,
            verbose=0 if 'background' in config.mode else 1)
