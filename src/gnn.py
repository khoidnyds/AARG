import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import networkx as nx
import logging

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.regularizers import l2
from spektral.data import Dataset, Graph
from spektral.layers import GCNConv
from spektral.data.loaders import SingleLoader


class GNN():
    def __init__(self, path_graph, params, out_dir):
        self.path_graph = path_graph
        self.params = params
        self.path_model = out_dir.joinpath("gnn")

    def train(self):
        G = nx.read_gpickle(self.path_graph)

        class MyDataset(Dataset):
            def read(self):
                return [Graph(a=nx.to_numpy_matrix(G),
                              e=np.array(
                                  [i for i in nx.get_edge_attributes(G, "features").values()]),
                              x=np.array(
                                  [i for i in nx.get_node_attributes(G, "features").values()]),
                              y=np.array([i for i in nx.get_node_attributes(G, "label").values()]))]
        dataset = MyDataset()

        mask_tr, mask_va = train_test_split(
            list(range(dataset.n_nodes)), test_size=0.3)
        mask_va, mask_te = train_test_split(mask_va, test_size=0.3)
        dataset.mask_tr = np.array(
            [True if i in mask_tr else False for i in range(dataset.n_nodes)])
        dataset.mask_va = np.array(
            [True if i in mask_va else False for i in range(dataset.n_nodes)])
        dataset.mask_te = np.array(
            [True if i in mask_te else False for i in range(dataset.n_nodes)])

        def mask_to_weights(mask):
            return mask.astype(np.float32) / np.count_nonzero(mask)

        weights_tr, weights_va, weights_te = (
            mask_to_weights(mask)
            for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
        )

        # Preprocessing operations
        A = GCNConv.preprocess(dataset[0].a).astype('f4')
        N, F = dataset[0].x.shape
        # Model definition
        X_in = Input(shape=(F, ))
        fltr_in = Input((N, ), sparse=True)

        dropout_1 = Dropout(self.params['dropout'])(X_in)
        graph_conv_1 = GCNConv(self.params['channels_1'],
                               activation='relu',
                               kernel_regularizer=l2(self.params['l2_reg']),
                               use_bias=False)([dropout_1, fltr_in])

        dropout_2 = Dropout(self.params['dropout'])(graph_conv_1)
        graph_conv_2 = GCNConv(self.params['channels_2'],
                               activation='relu',
                               kernel_regularizer=l2(self.params['l2_reg']),
                               use_bias=False)([dropout_2, fltr_in])

        dropout_3 = Dropout(self.params['dropout'])(graph_conv_2)
        graph_conv_3 = GCNConv(dataset[0].n_labels,
                               activation='relu',
                               kernel_regularizer=l2(self.params['l2_reg']),
                               use_bias=False)([dropout_3, fltr_in])

        output = tf.keras.activations.sigmoid(graph_conv_3)

        # Build model
        model = Model(inputs=[X_in, fltr_in], outputs=output)
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy')
        model.summary(print_fn=lambda x: logging.info(x))

        # Train model
        validation_data = ([dataset[0].x, A], dataset[0].y, weights_tr)
        model.fit([dataset[0].x, A], dataset[0].y, sample_weight=weights_va,
                  epochs=self.params['epochs'],
                  batch_size=N,
                  validation_data=validation_data,
                  shuffle=False,
                  callbacks=[
            EarlyStopping(patience=self.params['es_patience'],
                          restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir='./Tensorboard',)
        ])
        model.save(self.path_model)
        # Test model
        # loader_te = SingleLoader(dataset, sample_weights=weights_te)
        # eval_results = model.evaluate(
        #     loader_te.load(), steps=loader_te.steps_per_epoch)
        # logging.info(
        #     f"Test loss: {eval_results[0]} - Test accuracy: {eval_results[1]}")
        return self.path_model
