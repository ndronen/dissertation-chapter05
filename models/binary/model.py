from chapter05.dataset import BinaryModelDatasetGenerator

import sys
sys.setrecursionlimit(5000)

import numpy as np
import pandas as pd
import sklearn.neighbors

from keras.models import Sequential, Graph
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.normalization import BatchNormalization

import modeling.data
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights,
    build_hierarchical_softmax_layer)

from spelling.dictionary import (
        HashBucketRetriever,
        AspellRetriever,
        EditDistanceRetriever,
        NearestNeighborsRetriever,
        RetrieverCollection)
import spelling.features

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
    graph.add_input(config.real_word_input_name,
            input_shape=(config.model_input_width,), dtype='int')

    graph.add_shared_node(
        build_embedding_layer(config, input_width=config.model_input_width),
        name='embedding',
        inputs=[config.non_word_input_name, config.real_word_input_name],
        outputs=['non_word_embedding', 'real_word_embedding'])

    #graph.add_node(build_embedding_layer(config, input_width=config.model_input_width),
    #        name='non_word_embedding', input=config.non_word_input_name)
    graph.add_node(build_convolutional_layer(config), name='non_word_conv', input='non_word_embedding')
    non_word_prev_layer = add_bn_relu(graph, config, 'non_word_conv')
    graph.add_node(build_pooling_layer(config, input_width=config.model_input_width),
            name='non_word_pool', input=non_word_prev_layer)
    graph.add_node(Flatten(), name='non_word_flatten', input='non_word_pool')

    #graph.add_node(build_embedding_layer(config, input_width=config.model_input_width),
    #        name='real_word_embedding', input=config.real_word_input_name)
    graph.add_node(build_convolutional_layer(config), name='real_word_conv', input='real_word_embedding')
    real_word_prev_layer = add_bn_relu(graph, config, 'real_word_conv')
    graph.add_node(build_pooling_layer(config, input_width=config.model_input_width),
            name='real_word_pool', input=real_word_prev_layer)
    graph.add_node(Flatten(), name='real_word_flatten', input='real_word_pool')

    # Add some number of fully-connected layers without skip connections.
    prev_layer = None

    for i in range(config.n_fully_connected):
        layer_name = 'dense%02d' % i
        l = build_dense_layer(config, n_hidden=config.n_hidden)
        if i == 0:
            graph.add_node(l, name=layer_name,
                inputs=['non_word_flatten', 'real_word_flatten'])
        else:
            graph.add_node(l, name=layer_name, input=prev_layer)
        prev_layer = layer_name
        if config.batch_normalization:
            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
            prev_layer = layer_name+'bn'
        if config.dropout_fc_p > 0.:
            graph.add_node(Dropout(config.dropout_fc_p), name=layer_name+'do', input=prev_layer)
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
    
            l = build_dense_layer(config, n_hidden=config.n_hidden)
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

    graph.add_node(build_dense_layer(config, 2,
        activation='softmax'), name='softmax', input=prev_layer)

    graph.add_output(name=config.target_name, input='softmax')

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
                self.config.real_word_input_name: x,
                self.config.target_name: y_one_hot
                }
            sample_weights = { "binary_correction_target": np.array([0.4, 10.]) }

            yield (d, sample_weights)

def fit(config):
    df = pd.read_csv(config.non_word_csv, sep='\t', encoding='utf8')
    df = df[df.binary_target == 0]

    # The simplest retrievers.
    aspell_retriever = AspellRetriever()
    edit_distance_retreiver = EditDistanceRetriever(df.real_word.unique())
    hash_bucket_retriever = HashBucketRetriever(df.real_word.unique(), spelling.features.metaphone)

    # A bit more complex -- this one takes an estimator.
    estimator = sklearn.neighbors.NearestNeighbors(n_neighbors=10, metric='hamming', algorithm='auto')
    nearest_neighbors_retriever = NearestNeighborsRetriever(df.real_word.unique(), estimator)

    # Combine them.
    retriever = RetrieverCollection(retrievers=[
            aspell_retriever, hash_bucket_retriever,
            edit_distance_retreiver, nearest_neighbors_retriever
            ])

    train_generator = BinaryModelDatasetGenerator(df.word.tolist(), df.real_word.tolist(),
            retriever, config.model_input_width)
    validation_generator = BinaryModelDatasetGenerator(df.word.tolist(), df.real_word.tolist(),
            retriever, config.model_input_width)

    graph = build_model(config)

    graph.fit_generator(train_generator.generate(),
            samples_per_epoch=10000,
            nb_epoch=10000,
            nb_worker=10,
            validation_data=validation_generator.generate(),
            nb_val_samples=10000,
            nb_val_worker=10)
