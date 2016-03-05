from modeling.callbacks import DenseWeightNormCallback
from chapter05.dataset import MulticlassModelDatasetGenerator

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from spelling.utils import build_progressbar

import sys
sys.setrecursionlimit(5000)

import numpy as np
import pandas as pd

from keras.models import Sequential, Graph
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.constraints import maxnorm
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
import keras.callbacks

import modeling.data
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights,
    build_hierarchical_softmax_layer)
from modeling.utils import balanced_class_weights

from spelling.dictionary import AspellRetriever, EditDistanceRetriever

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

def build_model(config, n_classes):
    np.random.seed(config.seed)

    graph = Graph()

    graph.add_input(config.non_word_input_name,
            input_shape=(config.model_input_width,), dtype='int')
    graph.add_node(build_embedding_layer(config, input_width=config.model_input_width),
            name='non_word_embedding', input=config.non_word_input_name)
    graph.add_node(build_convolutional_layer(config), name='non_word_conv', input='non_word_embedding')
    non_word_prev_layer = add_bn_relu(graph, config, 'non_word_conv')
    graph.add_node(build_pooling_layer(config, input_width=config.model_input_width),
            name='non_word_pool', input=non_word_prev_layer)
    graph.add_node(Flatten(), name='non_word_flatten', input='non_word_pool')
    graph.add_node(GaussianNoise(0.01), name="non_word_noise", input="non_word_flatten")

    # Add some number of fully-connected layers without skip connections.
    prev_layer = 'non_word_noise'
    for i,n_hidden in enumerate(config.fully_connected):
        layer_name = 'dense%02d' %i
        l = build_dense_layer(config, n_hidden=n_hidden)
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
    
            graph.add_node(Dense(config.n_hidden_residual, W_constraint=maxnorm(2)),
                    name=layer_name, input=prev_layer)
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

    if hasattr(config, 'n_hsm_classes'):
        graph.add_node(build_hierarchical_softmax_layer(config),
            name='softmax', input=prev_layer)
    else:
        graph.add_node(Dense(n_classes, W_constraint=maxnorm(10)),
            name='softmax', input=prev_layer)
        prev_layer = 'softmax'
        if config.batch_normalization:
            graph.add_node(BatchNormalization(), name='softmax_bn', input='softmax')
            prev_layer = 'softmax_bn'
        graph.add_node(Activation('softmax'), name='softmax_activation', input=prev_layer)

    graph.add_output(name='multiclass_correction_target', input='softmax_activation')

    load_weights(config, graph)

    optimizer = build_optimizer(config)

    graph.compile(loss={'multiclass_correction_target': config.loss}, optimizer=optimizer)

    return graph


class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, config, generator, n_samples, dictionary, target_map):
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
        g = self.generator.generate(exhaustive=True, train=False)
        n_failed = 0
        while True:
            pbar.update(counter)
            try:
                next_batch = next(g)
            except StopIteration:
                break

            assert isinstance(next_batch, dict)

            # The dictionary's predictions.  Get these first, so we can
            # skip any that the dictionary doesn't have suggestions for.
            # This is to ensure that the evaluation occurs on even ground.
            non_words = next_batch['non_word']
            correct_words = next_batch['correct_word']
            failed = []
            for i,non_word in enumerate(non_words):
                suggestions = self.dictionary[str(non_word)]
                try:
                    suggestion = suggestions[0]
                    target = self.target_map[suggestion]
                    if target is None:
                        raise ValueError('target is None for %s => %s' % (non_word, suggestion))
                    y_hat_dictionary.append(target)
                except IndexError:
                    # I don't know what to do if the dictionary doesn't
                    # offer any suggestions.
                    failed.append(True)
                except KeyError as e:
                    # Or if we don't have a target for the suggested replacement.
                    failed.append(True)

            if any(failed):
                n_failed += len(failed)
                continue

            # The gold standard.
            targets = next_batch[self.config.target_name]
            y.append(np.argmax(targets, axis=1))

            # The model's predictions.
            pred = self.model.predict(next_batch, verbose=0)[self.config.target_name]
            y_hat.append(np.argmax(pred, axis=1))

            counter += len(targets)
            #if counter >= self.n_samples:
            #    print('%d >= %d - stopping loop' % (counter, self.n_samples))
            #    break

        pbar.finish()

        self.config.logger('\n%d dictionary lookups failed reporting results for %d examples\n' %
                    (n_failed, len(y)))

        self.config.logger('\n')
        self.config.logger('Dictionary')
        self.config.logger('accuracy %.04f F1 %0.4f' %
            (accuracy_score(y, y_hat_dictionary), f1_score(y, y_hat_dictionary, average='weighted')))

        self.config.logger('\n')
        self.config.logger('ConvNet')
        self.config.logger('accuracy %.04f F1 %0.4f\n' %
            (accuracy_score(y, y_hat), f1_score(y, y_hat, average='weighted')))
        self.config.logger('\n')

def build_callbacks(config, generator, n_samples, dictionary, target_map):
    callbacks = []
    mc = MetricsCallback(config, generator, n_samples, dictionary, target_map)
    wn = DenseWeightNormCallback(config)
    es = keras.callbacks.EarlyStopping(patience=config.patience, verbose=1)
    cp = keras.callbacks.ModelCheckpoint(filepath=config.model_path + 'model.h5')
    callbacks.extend([mc, wn, es, cp])
    return callbacks

def fit(config, callbacks=[]):
    df = pd.read_csv(config.non_word_csv, sep='\t', encoding='utf8')
    vocabulary = df.real_word.unique().tolist()
    df = df[df.binary_target == 0]

    # Select examples by frequency and length.
    frequencies = df.multiclass_correction_target.value_counts().to_dict()
    df['frequency'] = df.multiclass_correction_target.apply(lambda x: frequencies[x])
    df['len'] = df.word.apply(len)
    mask = (df.frequency >= config.min_frequency) & (df.len >= config.min_length) & (df.len <= config.max_length)
    df = df[mask]
    print(len(df), len(df.multiclass_correction_target.unique()))

    le = LabelEncoder()
    le.fit(df.real_word)

    fast_retriever = AspellRetriever()
    #slow_retriever = EditDistanceRetriever(vocabulary)

    df_train, df_other = train_test_split(df, train_size=0.8, random_state=config.seed)
    train_words = set(df_train.word)
    other_words = set(df_other.word)
    leaked_words = train_words.intersection(other_words)
    df_other = df_other[~df_other.word.isin(leaked_words)]
    df_valid, df_test = train_test_split(df_other, train_size=0.5, random_state=config.seed)

    print('train %d validation %d test %d' % (len(df_train), len(df_valid), len(df_test)))

    train_targets = le.transform(df_train.real_word).tolist()
    noise_word_target = max(train_targets) + 1
    n_classes = noise_word_target + 1

    target_map = dict(zip(df_train.real_word, train_targets))
    for word in vocabulary:
        if word not in target_map:
            target_map[word] = noise_word_target

    if config.use_contrasting_cases:
        train_data = MulticlassModelDatasetGenerator(
            df_train.word.tolist(),
            df_train.real_word.tolist(),
            train_targets,
            config.model_input_width,
            n_classes,
            batch_size=config.batch_size,
            noise_word_target=noise_word_target,
            retriever=fast_retriever)
    else:
        train_data = MulticlassModelDatasetGenerator(
            df_train.word.tolist(),
            df_train.real_word.tolist(),
            train_targets,
            config.model_input_width,
            n_classes,
            batch_size=config.batch_size)

    # Don't configure the validation data set to make noise examples.
    validation_data = MulticlassModelDatasetGenerator(
        # non_words
        df_valid.head(config.n_val_samples).word.tolist(),
        # corrections
        df_valid.head(config.n_val_samples).real_word.tolist(),
        # targets
        le.transform(df_valid.head(config.n_val_samples).real_word.tolist()),
        # model_input_width
        config.model_input_width,
        n_classes,
        batch_size=config.batch_size)
    
    class_weight_targets = train_targets
    if config.use_contrasting_cases:
        # Add one noise example for every real example.
        class_weight_targets += [noise_word_target] * len(train_targets)
    class_weight = modeling.utils.balanced_class_weights(
        class_weight_targets,
        n_classes,
        class_weight_exponent=config.class_weight_exponent)
    if not config.use_contrasting_cases:
        class_weight[noise_word_target] = 0

    print('n_classes %d' % n_classes)

    graph = build_model(config, n_classes)

    config.logger('model has %d parameters' % graph.count_params())

    callbacks = build_callbacks(config,
            validation_data,
            n_samples=config.n_val_samples,
            dictionary=fast_retriever,
            target_map=target_map)

    verbose = 2 if 'background' in config.mode else 1

    graph.fit_generator(train_data.generate(train=True),
            samples_per_epoch=config.samples_per_epoch,
            nb_worker=config.n_worker,
            nb_epoch=config.n_epoch,
            validation_data=validation_data.generate(train=False),
            #validation_data=train_data.generate(train=False),
            nb_val_samples=config.n_val_samples,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose)
