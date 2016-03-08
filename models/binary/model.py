from chapter05.dataset import BinaryModelDatasetGenerator, BinaryModelRealWordDatasetGenerator

import sys
sys.setrecursionlimit(5000)
import threading

import numpy as np
import pandas as pd
import sklearn.neighbors
from sklearn.cross_validation import train_test_split

from keras.models import Sequential, Graph
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.constraints import maxnorm
import keras.callbacks
from spelling.utils import build_progressbar

import modeling.data
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights,
    build_hierarchical_softmax_layer)
import modeling.callbacks 

import spelling.dictionary as spelldict
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

# Optionally add outputs to predict (1) jaro-winkler distance of
# non-word and real word and (2) the rank of the real word in
# a dictionary's ordered candidate list.
def add_linear_output_mlp(graph, prev_layer, name, input_dim, n_hidden, batch_normalization, loss):
    mlp = Sequential()
    for i in range(3):
        if i == 0:
            mlp.add(Dense(n_hidden, input_shape=(input_dim,)))
        else:
            mlp.add(Dense(n_hidden))
        if batch_normalization:
            mlp.add(BatchNormalization())
        #if i < 2:
        mlp.add(Activation('tanh'))
    mlp.add(Dense(1))
    #if batch_normalization:
    #    mlp.add(BatchNormalization())
    graph.add_node(mlp, name=name+'_dense', input=prev_layer)
    graph.add_output(name=name, input=name+'_dense')
    loss[name] = "mean_squared_error"

def add_linear_output_layer(graph, prev_layer, name, batch_normalization, loss):
    graph.add_node(Dense(1), name=name+'_dense', input=prev_layer)
    prev_layer = name+'_dense'
    if batch_normalization:
        graph.add_node(BatchNormalization(), 
                name=name+'_bn', input=prev_layer)
        prev_layer = name+'_bn'
    #graph.add_node(Activation('tanh'), name=name+'_tanh', input=prev_layer)
    graph.add_output(name=name, input=name+'_tanh')
    loss[name] = "mean_squared_error"

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

    # Add noise only to non-words.
    graph.add_node(GaussianNoise(config.gaussian_noise_sd),
            name='non_word_noise', input='non_word_embedding')

    graph.add_shared_node(
            build_convolutional_layer(config),
            name='conv',
            inputs=['non_word_noise', 'candidate_word_embedding'],
            outputs=['non_word_conv', 'candidate_word_conv'])

    non_word_prev_layer = add_bn_relu(graph, config, 'non_word_conv')
    graph.add_node(build_pooling_layer(config, input_width=config.model_input_width),
            name='non_word_pool', input=non_word_prev_layer)
    graph.add_node(Flatten(), name='non_word_flatten', input='non_word_pool')

    candidate_word_prev_layer = add_bn_relu(graph, config, 'candidate_word_conv')
    graph.add_node(build_pooling_layer(config, input_width=config.model_input_width),
            name='candidate_word_pool', input=candidate_word_prev_layer)
    graph.add_node(Flatten(), name='candidate_word_flatten', input='candidate_word_pool')

    # Add some number of fully-connected layers without skip connections.
    prev_layer = None

    if config.pool_merge_mode == 'cos':
        dot_axes = ([1], [1])
    else:
        dot_axes = -1

    for i,n_hidden in enumerate(config.fully_connected):
        layer_name = 'dense%02d' % i
        l = build_dense_layer(config, n_hidden=n_hidden)
        if i == 0:
            graph.add_node(l, name=layer_name,
                inputs=['non_word_flatten', 'candidate_word_flatten'],
                merge_mode=config.pool_merge_mode, dot_axes=dot_axes)
        else:
            graph.add_node(l, name=layer_name, input=prev_layer)
        prev_layer = layer_name
        if config.batch_normalization:
            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
            prev_layer = layer_name+'bn'
        graph.add_node(Activation('relu'),
                name=layer_name+'relu', input=prev_layer)
        if config.dropout_fc_p > 0.:
            graph.add_node(Dropout(config.dropout_fc_p), name=layer_name+'do', input=layer_name+'relu')
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

    # Save the name of the last dense layer for the distance and rank targets.
    last_dense_layer = prev_layer

    # Add softmax for binary prediction of whether the real word input
    # is the true correction for the non-word input.
    if len(config.fully_connected) == 0:
        graph.add_node(Dense(2, W_constraint=maxnorm(config.softmax_max_norm)),
                name='softmax',
                inputs=['non_word_flatten', 'candidate_word_flatten'],
                merge_mode=config.pool_merge_mode,
                dot_axes=dot_axes)
    else:
        graph.add_node(Dense(2, W_constraint=maxnorm(config.softmax_max_norm)),
                name='softmax', input=prev_layer)
    prev_layer = 'softmax'
    if config.batch_normalization:
        graph.add_node(BatchNormalization(), 
                name='softmax_bn', input='softmax')
        prev_layer = 'softmax_bn'
    graph.add_node(Activation('softmax'),
            name='softmax_activation', input=prev_layer)
    graph.add_output(name=config.target_name, input='softmax_activation')

    lossdict = {}
    lossdict[config.target_name] = config.loss

    for distance_name in config.distance_targets:
        add_linear_output_mlp(graph, 'dense00', distance_name+'_first',
                config.fully_connected[-1], 10, config.batch_normalization, lossdict)
        add_linear_output_mlp(graph, last_dense_layer, distance_name+'_last',
                config.fully_connected[-1], 10, config.batch_normalization, lossdict)

    if config.use_rank_target:
        add_linear_output_mlp(graph, 'dense00', 'candidate_rank_first', 
                config.fully_connected[-1], 10,
                config.batch_normalization, lossdict)
        add_linear_output_mlp(graph, last_dense_layer, 'candidate_rank_last', 
                config.fully_connected[-1], 10,
                config.batch_normalization, lossdict)

    load_weights(config, graph)

    optimizer = build_optimizer(config)

    graph.compile(loss=lossdict, optimizer=optimizer)

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
    def __init__(self, config, generator, n_samples, callbacks):
        self.__dict__.update(locals())
        del self.self

    def on_epoch_end(self, epoch, logs={}):
        correct = []
        y = []
        y_hat = []
        y_hat_binary = []
        y_hat_dictionary = []
        y_hat_dictionary_binary = []
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
            if targets[:, 1][np.argmax(pred[:, 1])] == 1:
                y_hat_binary.append(1)
            else:
                y_hat_binary.append(0)

            correct_word = d['correct_word'][0]

            y_hat_dictionary_tmp = []
            if d['candidate_word'][0] == correct_word:
                y_hat_dictionary_binary.append(1)
            else:
                y_hat_dictionary_binary.append(0)

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
        self.config.logger('Dictionary binary accuracy %.04f accuracy %.04f F1 %0.4f' % 
                (
                    sum(y_hat_dictionary_binary) / float(len(y_hat_dictionary_binary)),
                    accuracy_score(y, y_hat_dictionary),
                    f1_score(y, y_hat_dictionary))
                )
        self.config.logger('Dictionary confusion matrix')
        self.config.logger(confusion_matrix(y, y_hat_dictionary))

        model_binary_accuracy = sum(y_hat_binary) / float(len(y_hat_binary))
        model_accuracy = accuracy_score(y, y_hat)
        model_f1 = f1_score(y, y_hat)

        self.config.logger('\n')
        self.config.logger('ConvNet binary accuracy %.04f accuracy %.04f F1 %0.4f' % 
                (model_binary_accuracy, model_accuracy, model_f1))
        self.config.logger('ConvNet confusion matrix')
        self.config.logger(confusion_matrix(y, y_hat))

        self.config.logger('\n')

        logs['f1'] = model_f1
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)


def build_callbacks(config, generator, n_samples):
    # For this model, we want to monitor F1 for early stopping and
    # model checkpointing.  The way to do that is for the metrics callback
    # compute F1, put it in the logs dictionary that's passed to
    # on_epoch_end, and to pass that to the early stopping and model
    # checkpointing callbacks.
    controller_callbacks = []
    controller_callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=config.callback_monitor,
                mode=config.callback_monitor_mode,
                patience=config.patience,
                verbose=1))

    if 'persistent' in config.mode:
        controller_callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    monitor=config.callback_monitor,
                    mode=config.callback_monitor_mode,
                    filepath=config.model_path + 'model.h5',
                    save_best_only=True,
                    verbose=1))

    controller = MetricsCallback(config, generator, n_samples,
            callbacks=controller_callbacks)

    callbacks = []
    callbacks.append(controller)
    callbacks.append(modeling.callbacks.DenseWeightNormCallback(config))

    return callbacks

def fit(config):
    df = pd.read_csv(config.non_word_csv, sep='\t', encoding='utf8')
    df = df[df.binary_target == 0]

    # Select examples by frequency and length.
    frequencies = df.multiclass_correction_target.value_counts().to_dict()
    df['frequency'] = df.multiclass_correction_target.apply(lambda x: frequencies[x])
    df['len'] = df.word.apply(len)
    mask = (df.frequency >= config.min_frequency) & (df.len >= config.min_length) & (df.len <= config.max_length)
    df = df[mask]
    print(len(df), len(df.multiclass_correction_target.unique()))

    retriever_lock = threading.Lock()

    # Retrievers
    def build_retriever(vocabulary):
        with retriever_lock:
            aspell_retriever = spelldict.AspellRetriever()
            edit_distance_retriever = spelldict.EditDistanceRetriever(vocabulary)
            retriever = spelldict.RetrieverCollection([aspell_retriever, edit_distance_retriever])
            jaro_sorter = spelldict.DistanceSorter('jaro_winkler')
            return spelldict.SortingRetriever(retriever, jaro_sorter)

    df_train, df_other = train_test_split(df, train_size=0.8, random_state=config.seed)

    print('train %d other %d' % (len(df_train), len(df_other)))

    if config.real_words_only:
        train_words = set(df_train.real_word)
        other_words = set(df_other.real_word)
        leaked_words = train_words.intersection(other_words)
    else:
        train_words = set(df_train.word)
        other_words = set(df_other.word)
        leaked_words = train_words.intersection(other_words)
    df_other = df_other[~df_other.word.isin(leaked_words)]
    df_valid, df_test = train_test_split(df_other, train_size=0.5, random_state=config.seed)

    print('train %d validation %d test %d' % (len(df_train), len(df_valid), len(df_test)))
    training_freqs = df_train.real_word.value_counts()
    print('least frequent word', training_freqs.tail(1))

    vocabulary = df.real_word.unique().tolist()

    print('vocabulary %d' % len(vocabulary))

    if config.real_words_only:
        train_generator = BinaryModelRealWordDatasetGenerator(
                df_train.real_word.tolist(),
                config.model_input_width,
                distance_targets=config.distance_targets)

        validation_generator = BinaryModelRealWordDatasetGenerator(
                df_train.real_word.tolist(),
                config.model_input_width,
                distance_targets=config.distance_targets)

        validation_loss_generator = BinaryModelRealWordDatasetGenerator(
                df_train.real_word.tolist(),
                config.model_input_width,
                distance_targets=config.distance_targets)
    else:
        train_generator = BinaryModelDatasetGenerator(
                df_train.word.tolist(),
                df_train.real_word.tolist(),
                build_retriever(vocabulary),
                config.model_input_width,
                config.sample_weight_exponent,
                use_correct_word_as_non_word_example=config.use_correct_word_as_non_word_example,
                distance_targets=config.distance_targets,
                max_candidates=config.max_candidates)
    
        validation_generator = BinaryModelDatasetGenerator(
                df_valid.head(config.n_val_samples).word.tolist(),
                df_valid.head(config.n_val_samples).real_word.tolist(),
                build_retriever(vocabulary),
                config.model_input_width,
                config.sample_weight_exponent,
                distance_targets=config.distance_targets,
                max_candidates=config.max_candidates)
    
        validation_loss_generator = BinaryModelDatasetGenerator(
                df_valid.head(config.n_val_samples).word.tolist(),
                df_valid.head(config.n_val_samples).real_word.tolist(),
                build_retriever(vocabulary),
                config.model_input_width,
                config.sample_weight_exponent,
                distance_targets=config.distance_targets,
                max_candidates=config.max_candidates)

    graph = build_model(config)

    config.logger('model has %d parameters' % graph.count_params())

    callbacks = build_callbacks(config, validation_generator,
            n_samples=config.n_val_samples)

    verbose = 2 if 'background' in config.mode else 1

    graph.fit_generator(train_generator.generate(),
            samples_per_epoch=config.samples_per_epoch,
            nb_epoch=config.n_epoch,
            nb_worker=config.n_worker,
            validation_data=validation_loss_generator.generate(),
            nb_val_samples=config.n_val_samples,
            callbacks=callbacks,
            verbose=verbose)
