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


class GNN():
    def __init__(self, graph, params):
        self.graph = graph
        self.params = params

    def train(self):

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

        # Parameters
        channels = 16           # Number of channels in the first layer
        dropout = 0.5           # Dropout rate for the features
        l2_reg = 5e-4           # L2 regularization rate
        learning_rate = 1e-2    # Learning rate
        epochs = 200            # Number of training epochs
        es_patience = 10        # Patience for early stopping
        # Preprocessing operations
        A = GCNConv.preprocess(adj).astype('f4')
        F = len(graph_node_features[0])
        N = len(graph_node_features)
        # Model definition
        X_in = Input(shape=(F, ))
        fltr_in = Input((N, ), sparse=True)

        dropout_1 = Dropout(dropout)(X_in)
        graph_conv_1 = GCNConv(channels,
                               activation='relu',
                               kernel_regularizer=l2(l2_reg),
                               use_bias=False)([dropout_1, fltr_in])

        dropout_2 = Dropout(dropout)(graph_conv_1)
        graph_conv_2 = GCNConv(len(labels[0]),
                               activation='softmax',
                               use_bias=False)([dropout_2, fltr_in])

        # Build model
        model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      weighted_metrics=['acc'])
        model.summary()

        tbCallBack_GCN = tf.keras.callbacks.TensorBoard(
            log_dir='./Tensorboard_GCN_cora',
        )
        callback_GCN = [tbCallBack_GCN]

        # Train model
        validation_data = ([graph_node_features, A], labels, dataset.mask_va)
        model.fit([graph_node_features, A], labels, sample_weight=dataset.mask_tr,
                  epochs=epochs,
                  batch_size=N,
                  validation_data=validation_data,
                  shuffle=False,
                  callbacks=[
            EarlyStopping(patience=es_patience,
                          restore_best_weights=True),
            tbCallBack_GCN
        ])

    def test(self):
        pass
