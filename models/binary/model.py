from modeling.dataset import HDF5FileDataset

import sys
sys.setrecursionlimit(5000)

import numpy as np

from keras.models import Sequential, Graph
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.normalization import BatchNormalization

import modeling.data
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights,
    build_hierarchical_softmax_layer)

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

def add_bn_relu(graph, args, prev_layer):
    bn_name = prev_layer + '_bn'
    relu_name = prev_layer + '_relu'
    if args.batch_normalization:
        graph.add_node(BatchNormalization(), name=bn_name, input=prev_layer)
        prev_layer = bn_name
    graph.add_node(Activation('relu'), name=relu_name, input=prev_layer)
    return relu_name

def build_model(args, train_data, validation_data):
    np.random.seed(args.seed)

    graph = Graph()

    non_word_input = 'non_word_marked_chars'
    real_word_input = 'real_word_marked_chars'

    non_word_input_width = train_data.data[non_word_input].shape[1]
    real_word_input_width = train_data.data[real_word_input].shape[1]

    print('non_word_input_width', non_word_input_width)
    print('real_word_input_width', real_word_input_width)

    graph.add_input(non_word_input, input_shape=(non_word_input_width,), dtype='int')
    graph.add_node(build_embedding_layer(args, input_width=non_word_input_width),
            name='non_word_embedding', input=non_word_input)
    graph.add_node(build_convolutional_layer(args), name='non_word_conv', input='non_word_embedding')
    non_word_prev_layer = add_bn_relu(graph, args, 'non_word_conv')
    graph.add_node(build_pooling_layer(args, input_width=non_word_input_width),
            name='non_word_pool', input=non_word_prev_layer)
    graph.add_node(Flatten(), name='non_word_flatten', input='non_word_pool')

    graph.add_input(real_word_input, input_shape=(real_word_input_width,), dtype='int')
    graph.add_node(build_embedding_layer(args, input_width=real_word_input_width),
            name='real_word_embedding', input=real_word_input)
    graph.add_node(build_convolutional_layer(args), name='real_word_conv', input='real_word_embedding')
    real_word_prev_layer = add_bn_relu(graph, args, 'real_word_conv')
    graph.add_node(build_pooling_layer(args, input_width=real_word_input_width),
            name='real_word_pool', input=real_word_prev_layer)
    graph.add_node(Flatten(), name='real_word_flatten', input='real_word_pool')

    # Add some number of fully-connected layers without skip connections.
    prev_layer = 'join_non_and_real'
    for i in range(args.n_fully_connected):
        layer_name = 'dense%02d' %i
        l = build_dense_layer(args, n_hidden=args.n_hidden)
        if i == 0:
            graph.add_node(l, name=layer_name,
                inputs=['non_word_flatten', 'real_word_flatten'])
        else:
            graph.add_node(l, name=layer_name, input=prev_layer)
        prev_layer = layer_name
        if args.batch_normalization:
            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
            prev_layer = layer_name+'bn'
        if args.dropout_fc_p > 0.:
            graph.add_node(Dropout(args.dropout_fc_p), name=layer_name+'do', input=prev_layer)
            prev_layer = layer_name+'do'
    
    # Add sequence of residual blocks.
    for i in range(args.n_residual_blocks):
        # Add a fixed number of layers per residual block.
        block_name = '%02d' % i

        graph.add_node(Identity(), name=block_name+'input', input=prev_layer)
        prev_layer = block_input_layer = block_name+'input'

        try:
            n_layers_per_residual_block = args.n_layers_per_residual_block
        except AttributeError:
            n_layers_per_residual_block = 2

        for layer_num in range(n_layers_per_residual_block):
            layer_name = 'h%s%02d' % (block_name, layer_num)
    
            l = build_dense_layer(args, n_hidden=args.n_hidden)
            graph.add_node(l, name=layer_name, input=prev_layer)
            prev_layer = layer_name
    
            if args.batch_normalization:
                graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
                prev_layer = layer_name+'bn'
    
            if i < n_layers_per_residual_block:
                a = Activation('relu')
                graph.add_node(Activation('relu'), name=layer_name+'relu', input=prev_layer)
                prev_layer = layer_name+'relu'
                if args.dropout_fc_p > 0.:
                    graph.add_node(Dropout(args.dropout_fc_p), name=layer_name+'do', input=prev_layer)
                    prev_layer = layer_name+'do'

        graph.add_node(Identity(), name=block_name+'output', inputs=[block_input_layer, prev_layer], merge_mode='sum')
        graph.add_node(Activation('relu'), name=block_name+'relu', input=block_name+'output')
        prev_layer = block_input_layer = block_name+'relu'

    #if hasattr(args, 'n_hsm_classes'):
    #    graph.add_node(build_hierarchical_softmax_layer(args),
    #            name='softmax', input=prev_layer)
    #else:

    graph.add_node(build_dense_layer(args, 2,
        activation='softmax'), name='softmax', input=prev_layer)

    graph.add_output(name='binary_target', input='softmax')

    load_weights(args, graph)

    optimizer = build_optimizer(args)

    graph.compile(loss={'binary_target': args.loss}, optimizer=optimizer)

    return graph


def fit(config, callbacks=[]):
    train_data = HDF5FileDataset(
        config.train_path,
        config.data_name,
        [config.target_name],
        config.batch_size,
        config.seed)

    validation_data = HDF5FileDataset(
        config.validation_path,
        config.data_name,
        [config.target_name],
        config.batch_size,
        config.seed)

    graph = build_model(config, train_data)

    class_weight = train_data.class_weights(
        config.class_weight_exponent,
        config.target_name)

    graph.fit_generator(train_data.generate(),
            samples_per_epoch=int(train_data.n/100),
            nb_epoch=args.n_epochs,
            validation_data=validation_data.get_dict(),
            callbacks=callbacks,
            class_weight=train_data.class_weights(args.class_weight_exponent))
