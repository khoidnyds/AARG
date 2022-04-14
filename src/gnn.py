from pyfaidx import Fasta
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import scipy.sparse as sp

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from spektral.models.gnn_explainer import GNNExplainer
from spektral.transforms import LayerPreprocess
from spektral.transforms.normalize_adj import NormalizeAdj
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.regularizers import l2

from spektral.layers import GCNConv, GlobalSumPool

from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import networkx as nx


class GNN():
    def __init__(self, path_graph, params, out_dir):
        self.path_graph = path_graph
        self.params = params
        self.model = out_dir.joinpath("gnn.model")

    def train(self):
        G = nx.read_gpickle(self.path_graph)

        graph = Graph(a=nx.to_numpy_matrix(G),
                      e=np.array(
                          [i for i in nx.get_edge_attributes(G, "features").values()]),
                      x=np.array(
                          [i for i in nx.get_node_attributes(G, "features").values()]),
                      y=np.array([i for i in nx.get_node_attributes(G, "label").values()]))

        mask_tr, mask_va = train_test_split(
            list(range(graph.n_nodes)), test_size=0.3)
        mask_va, mask_te = train_test_split(mask_va, test_size=0.3)
        graph.mask_tr = np.array(
            [True if i in mask_tr else False for i in range(graph.n_nodes)])
        graph.mask_va = np.array(
            [True if i in mask_va else False for i in range(graph.n_nodes)])
        graph.mask_te = np.array(
            [True if i in mask_te else False for i in range(graph.n_nodes)])

        def mask_to_weights(mask):
            return mask.astype(np.float32) / np.count_nonzero(mask)

        weights_tr, weights_va, weights_te = (
            mask_to_weights(mask)
            for mask in (graph.mask_tr, graph.mask_va, graph.mask_te)
        )

        # Preprocessing operations
        A = GCNConv.preprocess(graph.a).astype('f4')
        F = len(graph.x[0])
        N = len(graph.x)
        # Model definition
        X_in = Input(shape=(F, ))
        fltr_in = Input((N, ), sparse=True)

        dropout_1 = Dropout(self.params['dropout'])(X_in)
        graph_conv_1 = GCNConv(self.params['channels'],
                               activation='relu',
                               kernel_regularizer=l2(self.params['l2_reg']),
                               use_bias=False)([dropout_1, fltr_in])

        dropout_2 = Dropout(self.params['dropout'])(graph_conv_1)
        graph_conv_2 = GCNConv(graph.n_labels,
                               activation='softmax',
                               use_bias=False)([dropout_2, fltr_in])

        # Build model
        model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      weighted_metrics=['acc'])
        model.summary()

        tbCallBack_GCN = tf.keras.callbacks.TensorBoard(
            log_dir='./Tensorboard_GCN_cora',
        )
        callback_GCN = [tbCallBack_GCN]

        # Train model
        validation_data = ([graph.x, A], graph.y, graph.mask_va)
        model.fit([graph.x, A], graph.y, sample_weight=graph.mask_tr,
                  epochs=self.params['epochs'],
                  batch_size=N,
                  validation_data=validation_data,
                  shuffle=False,
                  callbacks=[
            EarlyStopping(patience=self.params['es_patience'],
                          restore_best_weights=True),
            tbCallBack_GCN
        ])
        return self.model

    def test(self):
        pass
