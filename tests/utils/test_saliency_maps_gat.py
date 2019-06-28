# -*- coding: utf-8 -*-
#
# Copyright 2018-2019 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import stellargraph.utils.saliency_maps_gat as saliency_gat
import numpy as np
from stellargraph.layer import GraphAttention
from stellargraph import StellarGraph
from stellargraph.layer import GAT
from stellargraph.mapper import FullBatchNodeGenerator
from keras import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import networkx as nx
import keras.backend as K
import keras


def example_graph_1(feature_size=None):
    G = nx.Graph()
    elist = [(0, 1), (0, 2), (2, 3), (3, 4), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    G.add_nodes_from([0, 1, 2, 3, 4], label="default")
    G.add_edges_from(elist, label="default")

    # Add example features
    if feature_size is not None:
        for v in G.nodes():
            G.node[v]["feature"] = np.ones(feature_size)
        return StellarGraph(G, node_features="feature")

    else:
        return StellarGraph(G)


def create_GAT_model(graph):
    generator = FullBatchNodeGenerator(graph, sparse=False)
    train_gen = generator.flow([0, 1], np.array([[1, 0], [0, 1]]))

    gat = GAT(
        layer_sizes=[2, 2],
        generator=generator,
        bias=False,
        in_dropout=0,
        attn_dropout=0,
        activations=["elu", "softmax"],
        normalize=None,
        saliency_map_support=True,
    )
    for layer in gat._layers:
        layer._initializer = "ones"
    x_inp, x_out = gat.node_model()
    keras_model = Model(inputs=x_inp, outputs=x_out)
    return gat, keras_model, generator, train_gen


def get_ego_node_num(graph, target_idx):
    G_ego = nx.ego_graph(graph, target_idx, radius=2)
    return G_ego.number_of_nodes()


def test_ig_saliency_map():
    graph = example_graph_1(feature_size=4)
    base_model, keras_model_gat, generator, train_gen = create_GAT_model(graph)
    keras_model_gat.compile(
        optimizer=Adam(lr=0.1), loss=categorical_crossentropy, weighted_metrics=["acc"]
    )
    weights = [
        np.array(
            [
                [0.47567585, 0.7989239],
                [0.33588523, 0.19814175],
                [0.15685713, 0.43643117],
                [0.7725941, 0.68441933],
            ]
        ),
        np.array([[0.71832293], [0.8542117]]),
        np.array([[0.46560588], [0.8165422]]),
        0.32935917,
        0.29731724,
        np.array([[0.4391179, 0.595691], [0.06000895, 0.2613866]]),
        np.array([[0.43496376], [0.02840129]]),
        np.array([[0.33972418], [0.22352563]]),
        1.0,
        0.0,
    ]
    keras_model_gat.set_weights(weights)
    for var in keras_model_gat.non_trainable_weights:
        if "ig_delta" in var.name:
            K.set_value(var, 1)
        if "ig_non_exist_edge" in var.name:
            K.set_value(var, 0)

    ig_saliency = saliency_gat.IntegratedGradients(keras_model_gat, train_gen)
    target_idx = 0
    class_of_interest = 0
    ig_link_importance = ig_saliency.get_integrated_link_masks(
        target_idx, class_of_interest, steps=200
    )
    ig_link_importance_ref = np.array(
        [
            [3.99e-08, 3.99e-08, 3.99e-08, 0, 0],
            [-3.10e-08, -3.10e-08, 0, 0, 0],
            [3.14e-08, 0.00e00, 3.14e-08, 3.14e-08, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert pytest.approx(ig_link_importance_ref, ig_link_importance, abs=1e-10)
    non_zero_edge_importance = np.count_nonzero(ig_link_importance)
    assert 8 == non_zero_edge_importance
    ig_node_importance = ig_saliency.get_node_importance(
        target_idx, class_of_interest, steps=200
    )
    assert pytest.approx(ig_node_importance, np.array([-13.06, -9.32, -7.46, -3.73, 0]))
    non_zero_node_importance = np.count_nonzero(ig_node_importance)
    assert 4 == non_zero_node_importance
