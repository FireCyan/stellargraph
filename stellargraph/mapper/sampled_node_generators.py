# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mappers to provide input data for the graph models in layers.

"""
__all__ = [
    "GraphSAGENodeGenerator",
    "HinSAGENodeGenerator",
    "Attri2VecNodeGenerator",
    "Node2VecNodeGenerator",
    "DirectedGraphSAGENodeGenerator",
]

import warnings
import operator
import random
import abc
import warnings
import numpy as np
import itertools as it
import networkx as nx
import scipy.sparse as sps
from tensorflow.keras import backend as K
from functools import reduce
from tensorflow.keras.utils import Sequence
from collections import defaultdict

from ..data import (
    SampledBreadthFirstWalk,
    SampledHeterogeneousBreadthFirstWalk,
    DirectedBreadthFirstNeighbours,
)
from ..core.graph import StellarGraph, GraphSchema
from ..core.utils import is_real_iterable
from ..core.validation import comma_sep
from . import NodeSequence, Generator
from ..random import SeededPerBatch


class BatchedNodeGenerator(Generator):
    """
    Abstract base class for graph data generators.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning.

    Do not use this base class: use a subclass specific to the method.

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        schema (GraphSchema): [Optional] Schema for the graph, for heterogeneous graphs.
    """

    def __init__(self, G, batch_size, schema=None, use_node_features=True):
        if not isinstance(G, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph object.")

        self.graph = G
        self.batch_size = batch_size

        # This is a node generator and requries a model with one root nodes per query
        self.multiplicity = 1

        # We need a schema for compatibility with HinSAGE
        if schema is None:
            self.schema = G.create_graph_schema()
        elif isinstance(schema, GraphSchema):
            self.schema = schema
        else:
            raise TypeError("Schema must be a GraphSchema object")

        # We will need real node types here
        self.head_node_types = None

        # Create sampler for GraphSAGE
        self.sampler = None

        # Check if the graph has features
        if use_node_features:
            G.check_graph_for_ml()

    @abc.abstractmethod
    def sample_features(self, head_nodes, batch_num):
        pass

    def num_batch_dims(self):
        return 1

    def flow(self, node_ids, targets=None, shuffle=False, seed=None):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        The node IDs are the nodes to train or inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset of the nodes in the graph.

        The targets are an array of numeric targets corresponding to the
        supplied node_ids to be used by the downstream task. They should
        be given in the same order as the list of node IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downstream task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            node_ids: an iterable of node IDs
            targets: a 2D array of numeric targets with shape
                ``(len(node_ids), target_size)``
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with with StellarGraph models
            in Keras methods ``fit``, ``evaluate``,
            and ``predict``

        """
        if self.head_node_types is not None:
            expected_node_type = self.head_node_types[0]
        else:
            expected_node_type = None

        node_ilocs = self.graph.node_ids_to_ilocs(node_ids)
        node_types = self.graph.node_type(node_ilocs, use_ilocs=True)
        invalid = node_ilocs[node_types != expected_node_type]

        if len(invalid) > 0:
            raise ValueError(
                f"node_ids: expected all nodes to be of type {expected_node_type}, "
                f"found some nodes with wrong type: {comma_sep(invalid, stringify=format)}"
            )

        return NodeSequence(
            self.sample_features,
            self.batch_size,
            node_ilocs,
            targets,
            shuffle=shuffle,
            seed=seed,
        )

    def flow_from_dataframe(self, node_targets, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_targets: a Pandas DataFrame of numeric targets indexed
                by the node ID for that target.
            shuffle (bool): If True the node_ids will be shuffled at each
                epoch, if False the node_ids will be processed in order.

        Returns:
            A NodeSequence object to use with with StellarGraph models
            in Keras methods ``fit``, ``evaluate``,
            and ``predict``

        """
        return self.flow(node_targets.index, node_targets.values, shuffle=shuffle)


class GraphSAGENodeGenerator(BatchedNodeGenerator):
    """
    A data generator for node prediction with Homogeneous GraphSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object with node features.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = GraphSAGENodeGenerator(G, 50, [10,10])
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    .. seealso::

       Model using this generator: :class:`.GraphSAGE`.

       Some examples using this generator (see the model for more):

       - `node classification <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/graphsage-node-classification.html>`__
       - `unsupervised representation learning via Deep Graph Infomax <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/deep-graph-infomax-embeddings.html>`__

       Related functionality:

       - :class:`.Neo4jGraphSAGENodeGenerator` for using :class:`.GraphSAGE` with Neo4j
       - :class:`.CorruptedGenerator` for unsupervised training using :class:`.DeepGraphInfomax`
       - :class:`.GraphSAGELinkGenerator` for link prediction, unsupervised training using random walks and related tasks
       - :class:`.DirectedGraphSAGENodeGenerator` for directed graphs
       - :class:`.HinSAGENodeGenerator` for heterogeneous graphs

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        num_samples (list): The number of samples per layer (hop) to take.
        seed (int): [Optional] Random seed for the node sampler.
        weighted (bool, optional): If True, sample neighbours using the edge weights in the graph.
    """

    def __init__(
        self, G, batch_size, num_samples, seed=None, name=None, weighted=False
    ):
        super().__init__(G, batch_size)

        self.num_samples = num_samples
        self.head_node_types = self.schema.node_types
        self.name = name
        self.weighted = weighted

        # Check that there is only a single node type for GraphSAGE
        if len(self.head_node_types) > 1:
            warnings.warn(
                "running homogeneous GraphSAGE on a graph with multiple node types",
                RuntimeWarning,
                stacklevel=2,
            )

        # Create sampler for GraphSAGE
        self._samplers = SeededPerBatch(
            lambda s: SampledBreadthFirstWalk(G, graph_schema=self.schema, seed=s),
            seed=seed,
        )

    def sample_features(self, head_nodes, batch_num):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            batch_num (int): Batch number

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where ``num_sampled_at_layer`` is the cumulative product of ``num_samples``
            for that layer.
        """
        node_samples = self._samplers[batch_num].run(
            nodes=head_nodes, n=1, n_size=self.num_samples, weighted=self.weighted
        )

        # The number of samples for each head node (not including itself)
        num_full_samples = np.sum(np.cumprod(self.num_samples))

        # Reshape node samples to sensible format
        def get_levels(loc, lsize, samples_per_hop, walks):
            end_loc = loc + lsize
            walks_at_level = list(it.chain(*[w[loc:end_loc] for w in walks]))
            if len(samples_per_hop) < 1:
                return [walks_at_level]
            return [walks_at_level] + get_levels(
                end_loc, lsize * samples_per_hop[0], samples_per_hop[1:], walks
            )

        nodes_per_hop = get_levels(0, 1, self.num_samples, node_samples)
        node_type = self.head_node_types[0]

        # Get features for sampled nodes
        batch_feats = [
            self.graph.node_features(layer_nodes, node_type, use_ilocs=True)
            for layer_nodes in nodes_per_hop
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [
            np.reshape(a, (len(head_nodes), -1 if np.size(a) > 0 else 0, a.shape[1]))
            for a in batch_feats
        ]
        return batch_feats

    def default_corrupt_input_index_groups(self):
        # everything can be shuffled together
        return [list(range(len(self.num_samples) + 1))]


class DirectedGraphSAGENodeGenerator(BatchedNodeGenerator):
    """
    A data generator for node prediction with homogeneous GraphSAGE models
    on directed graphs.

    At minimum, supply the StellarDiGraph, the batch size, and the number of
    node samples (separately for in-nodes and out-nodes)
    for each layer of the GraphSAGE model.

    The supplied graph should be a StellarDiGraph object with node features.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Example::

        G_generator = DirectedGraphSAGENodeGenerator(G, 50, [10,5], [5,1])
        train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
        test_data_gen = G_generator.flow(test_node_ids)

    .. seealso::

       Model using this generator: :class:`.DirectedGraphSAGE`.

       Example using this generator: `node classification <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/directed-graphsage-node-classification.html>`__.

       Related functionality:

       - :class:`.Neo4jDirectedGraphSAGENodeGenerator` for using :class:`.DirectedGraphSAGE` with Neo4j
       - :class:`.CorruptedGenerator` for unsupervised training using :class:`.DeepGraphInfomax`
       - :class:`.DirectedGraphSAGELinkGenerator` for link prediction and related tasks
       - :class:`.GraphSAGENodeGenerator` for undirected graphs
       - :class:`.HinSAGENodeGenerator` for heterogeneous graphs

    Args:
        G (StellarDiGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        in_samples (list): The number of in-node samples per layer (hop) to take.
        out_samples (list): The number of out-node samples per layer (hop) to take.
        seed (int): [Optional] Random seed for the node sampler.
        weighted (bool, optional): If True, sample neighbours using the edge weights in the graph.
    """

    def __init__(
        self,
        G,
        batch_size,
        in_samples,
        out_samples,
        seed=None,
        name=None,
        weighted=False,
    ):
        super().__init__(G, batch_size)

        # TODO Add checks for in- and out-nodes sizes
        self.in_samples = in_samples
        self.out_samples = out_samples
        self.head_node_types = self.schema.node_types
        self.name = name
        self.weighted = weighted

        # Check that there is only a single node type for GraphSAGE
        if len(self.head_node_types) > 1:
            warnings.warn(
                "running homogeneous GraphSAGE on a graph with multiple node types",
                RuntimeWarning,
                stacklevel=2,
            )

        # Create sampler for GraphSAGE
        self.sampler = DirectedBreadthFirstNeighbours(
            G, graph_schema=self.schema, seed=seed
        )

    def _max_slots(self):
        max_hops = len(self.in_samples)
        return 2 ** (max_hops + 1) - 1

    def sample_features(self, head_nodes, batch_num):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            batch_num (int): Batch number

        Returns:
            A list of feature tensors from the sampled nodes at each layer, each of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where ``num_sampled_at_layer`` is the total number (cumulative product)
            of nodes sampled at the given number of hops from each head node,
            given the sequence of in/out directions.
        """
        node_samples = self.sampler.run(
            nodes=head_nodes,
            n=1,
            in_size=self.in_samples,
            out_size=self.out_samples,
            weighted=self.weighted,
        )

        # Reshape node samples to sensible format
        # Each 'slot' represents the list of nodes sampled from some neighbourhood, and will have a corresponding
        # NN input layer. Every hop potentially generates both in-nodes and out-nodes, held separately,
        # and thus the slot (or directed hop sequence) structure forms a binary tree.

        node_type = self.head_node_types[0]

        max_slots = self._max_slots()
        features = [None] * max_slots  # flattened binary tree

        for slot in range(max_slots):
            nodes_in_slot = list(it.chain(*[sample[slot] for sample in node_samples]))
            features_for_slot = self.graph.node_features(
                nodes_in_slot, node_type, use_ilocs=True
            )
            resize = -1 if np.size(features_for_slot) > 0 else 0
            features[slot] = np.reshape(
                features_for_slot, (len(head_nodes), resize, features_for_slot.shape[1])
            )

        return features

    def default_corrupt_input_index_groups(self):
        # everything can be shuffled together
        return [list(range(self._max_slots()))]


class HinSAGENodeGenerator(BatchedNodeGenerator):
    """Keras-compatible data mapper for Heterogeneous GraphSAGE (HinSAGE)

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the HinSAGE model.

    The supplied graph should be a StellarGraph object with node features for all node types.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    Note that the shuffle argument should be True for training and
    False for prediction.

    .. seealso::

       Model using this generator: :class:`.HinSAGE`.

       Example using this generator: `unsupervised representation learning via Deep Graph Infomax <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/deep-graph-infomax-embeddings.html>`_.

       Related functionality:

       - :class:`.CorruptedGenerator` for unsupervised training using :class:`.DeepGraphInfomax`
       - :class:`.HinSAGELinkGenerator` for link prediction and related tasks
       - :class:`.GraphSAGENodeGenerator` for homogeneous graphs
       - :class:`.DirectedGraphSAGENodeGenerator` for directed homogeneous graphs

    Args:
        G (StellarGraph): The machine-learning ready graph
        batch_size (int): Size of batch to return
        num_samples (list): The number of samples per layer (hop) to take
        head_node_type (str, optional): The node type that will be given to the generator using the
            `flow` method, the model will expect this node type. This does not need to be specified
            if ``G`` has only one node type.
        schema (GraphSchema, optional): Graph schema for G.
        seed (int, optional): Random seed for the node sampler

    Example::

         G_generator = HinSAGENodeGenerator(G, 50, [10,10])
         train_data_gen = G_generator.flow(train_node_ids, train_node_labels)
         test_data_gen = G_generator.flow(test_node_ids)

     """

    def __init__(
        self,
        G,
        batch_size,
        num_samples,
        head_node_type=None,
        schema=None,
        seed=None,
        name=None,
        weighted_feat=False ##### Added by John #####
    ):
        super().__init__(G, batch_size, schema=schema)

        self.num_samples = num_samples
        self.name = name
        self.weighted_feat = weighted_feat ##### Added by John #####

        # The head node type
        if head_node_type is None:
            # infer the head node type, if this is a homogeneous-node graph
            head_node_type = G.unique_node_type(
                "head_node_type: expected a head node type because G has more than one node type, found node types: %(found)s"
            )

        if head_node_type not in self.schema.node_types:
            raise KeyError("Supplied head node type must exist in the graph")
        self.head_node_types = [head_node_type]

        # Create sampling schema
        self._sampling_schema = self.schema.sampling_layout(
            self.head_node_types, self.num_samples
        )
        self._type_adjacency_list = self.schema.type_adjacency_list(
            self.head_node_types, len(self.num_samples)
        )

        # Create sampler for HinSAGE
        self.sampler = SampledHeterogeneousBreadthFirstWalk(
            G, graph_schema=self.schema, seed=seed
        )

    ##### Added by John #####
    def _get_weight(self, head_node, list_node_samples, use_ilocs=False):
        """
        Get the weights (which is the edge weight, i.e. rating; number of bought per user-item pair) from the head node to all its sampled nodes
        Args:
            head_node: the source node to get the edge weights wtih
            list_node_samples: the list of destination nodes to get the edge weights with
        Returns:
            A list of weights between the source head node and the detination nodes            
        """
        list_weight = []
        list_node_loaded = [] # Keep a record of nodes that have appeared already, and make the weight to be 0 (effectively not re-counting the same node)
        for node in list_node_samples:
            # node == -1 means there are no neighbour nodes, i.e., the head node does not have any connection
            # if node has appeared before, then turn the weight to 0 too
            if (node == -1) or (node in list_node_loaded):
                list_weight.append(0)
            else:
                # A node may be sampled several times. To adjust that, divide by the total number of the node in the list_node_samples
                

                # TODO: divide by the number of nodes occuring?
                # list_weight.append(self.graph._edge_weights(head_node, node, use_ilocs=use_ilocs)[0]/list_node_samples.count(node))
                list_weight.append(self.graph._edge_weights(head_node, node, use_ilocs=use_ilocs)[0])
                list_node_loaded.append(node)

        # Divide by the total weight
        # total_weight = sum(list_weight)/len(list_weight)
        # if not total_weight == 0:
        #     list_weight = [x/total_weight for x in list_weight]
            
        return list_weight
        

    def _get_features(self, node_samples, head_size, use_ilocs=False):
        """
        Collect features from sampled nodes.
        Args:
            node_samples: A list of lists of node IDs
            head_size: The number of head nodes (typically the batch size).

        Returns:
            A list of numpy arrays that store the features for each head node.
        """
        # Note the if there are no samples for a node a zero array is returned.
        # Resize features to (batch_size, n_neighbours, feature_size)
        # for each node type (note that we can have different feature size for each node type)
        
        ##### Added by John #####
        """
        node_samples is a little complex. Will use the BWS project and num_sample = [8, 8] to do the coding.
        That means, to make this part compatible with our num_sample, need to adapt the code further in the future
        node_samples[0] to node_samples[2] represent features for different node layer (and different node types)
        (also note that node_samples[n][0] is the node type, node_samples[n][1] is the actual list of nodes)
            0: 0th layer for head node. Dimension = head_size
            1: 1st layer of head node's neighbour nodes. Given num_sample[0] = 8, there would be 8 neighbours for each head node (Dimension = head_size x num_sample[0])
            2: 2nd layer of head node's neighbours' neighbour nodes. Given num_sample[1] = 8, there would be 8x8 neighbours for each head node (Dimension = head_size x num_sample[0] x num_samples[1])
        """
        if self.weighted_feat:
            batch_feats = []
            
            for i in range(len(node_samples)):
                temp_batch_feats = []
                nt = node_samples[i][0]
                layer_nodes = node_samples[i][1]
                if i == 0: # head node. No weights required
                    batch_feats.append(self.graph.node_features(layer_nodes, nt, use_ilocs=use_ilocs))
                    head_nodes = layer_nodes # used for i == 2
                elif i in (1, 2): # 1st layer head node neighbours
                    # For loop through each user head node
                    if i == 1:
                        temp_head_nodes = head_nodes
                        layer_num = self.num_samples[0]
                        nb_1st_head_nodes = layer_nodes # used for i == 4
                    elif i == 2:
                        temp_head_nodes = nb_1st_head_nodes
                        layer_num = self.num_samples[1]
                    
                    for j in range(len(temp_head_nodes)):
                        h_node = temp_head_nodes[j]
                        list_temp_sampled = layer_nodes[(j*layer_num):((j+1)*layer_num)]
                        list_weight = self._get_weight(h_node, list_temp_sampled, use_ilocs=use_ilocs)

                        # Get the features from the list_temp_sampled
                        temp_feat = self.graph.node_features(list_temp_sampled, nt, use_ilocs=use_ilocs)
                        n_feat = temp_feat.shape[1]
                        # Broadcast the weight of the node for multiplication
                        list_weight_broadcast = np.repeat(list_weight, n_feat).reshape(layer_num, n_feat)

                        temp_batch_feats.extend(list_weight_broadcast*temp_feat)

                    batch_feats.append(np.array(temp_batch_feats))
            
        else:        
            batch_feats = [
                self.graph.node_features(layer_nodes, nt, use_ilocs=use_ilocs) # nt = node type
                for nt, layer_nodes in node_samples
            ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [np.reshape(a, (head_size, -1 if np.size(a) > 0 else 0, a.shape[1])) for a in batch_feats]

        return batch_feats

    def sample_features(self, head_nodes, batch_num):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            batch_num (int): Batch number

        Returns:
            A list of the same length as ``num_samples`` of collected features from
            the sampled nodes of shape:
            ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where ``num_sampled_at_layer`` is the cumulative product of ``num_samples``
            for that layer.
        """
        # Get sampled nodes
        node_samples = self.sampler.run(nodes=head_nodes, n=1, n_size=self.num_samples)

        # Reshape node samples to the required format for the HinSAGE model
        # This requires grouping the sampled nodes by edge type and in order
        nodes_by_type = [
            (
                nt,
                reduce(
                    operator.concat,
                    (samples[ks] for samples in node_samples for ks in indices),
                    [],
                ),
            )
            for nt, indices in self._sampling_schema[0]
        ]


        ##### Added by John #####
        batch_feats = self._get_features(nodes_by_type, len(head_nodes), use_ilocs=True)

        # Get features
        # batch_feats = [
        #     self.graph.node_features(layer_nodes, nt, use_ilocs=True)
        #     for nt, layer_nodes in nodes_by_type
        # ]

        # # Resize features to (batch_size, n_neighbours, feature_size)
        # batch_feats = [
        #     np.reshape(a, (len(head_nodes), -1 if np.size(a) > 0 else 0, a.shape[1]))
        #     for a in batch_feats
        # ]

        return batch_feats

    def default_corrupt_input_index_groups(self):
        # every sample of a given node type can be grouped together
        indices_per_nt = defaultdict(list)
        for tensor_idx, (nt, _) in enumerate(self._sampling_schema[0]):
            indices_per_nt[nt].append(tensor_idx)

        # ensure there's a consistent order both within each group, and across groups, ensure the
        # shuffling is deterministic (at least with respect to the model)
        return sorted(sorted(idx) for idx in indices_per_nt.values())


class Attri2VecNodeGenerator(BatchedNodeGenerator):
    """
    A node feature generator for node representation prediction with the
    attri2vec model.

    At minimum, supply the StellarGraph and the batch size.

    The supplied graph should be a StellarGraph object with node features.

    Use the :meth:`flow` method supplying the nodes to get an object
    that can be used as a Keras data generator.

    Example::

        G_generator = Attri2VecNodeGenerator(G, 50)
        data_gen = G_generator.flow(node_ids)

    .. seealso::

       Model using this generator: :class:`.Attri2Vec`.

       An example using this generator (see the model for more): `node classification <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/attri2vec-node-classification.html>`__.

       Related functionality: :class:`.Attri2VecLinkGenerator` for training, link prediction and related tasks.

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        name (str or None): Name of the generator (optional).
    """

    def __init__(self, G, batch_size, name=None):
        super().__init__(G, batch_size)
        self.name = name

    def sample_features(self, head_nodes, batch_num):
        """
        Sample content features of the head nodes, and return these as a list of feature
        arrays for the attri2vec algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.
            batch_num (int): Batch number

        Returns:
            A list of feature arrays, with each element being the feature of a
            head node.
        """

        batch_feats = self.graph.node_features(head_nodes, use_ilocs=True)
        return batch_feats

    def flow(self, node_ids):
        """
        Creates a generator/sequence object for node representation prediction
        with the supplied node ids.

        The node IDs are the nodes to inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset/all of the nodes in the graph.

        Args:
            node_ids: an iterable of node IDs.

        Returns:
            A NodeSequence object to use with the Attri2Vec model
            in the Keras method ``predict``.

        """
        node_ilocs = self.graph.node_ids_to_ilocs(node_ids)
        return NodeSequence(
            self.sample_features, self.batch_size, node_ilocs, shuffle=False
        )

    def flow_from_dataframe(self, node_ids):
        """
        Creates a generator/sequence object for node representation prediction
        by using the index of the supplied dataframe as the node ids.

        Args:
            node_ids: a Pandas DataFrame of node_ids.

        Returns:
            A NodeSequence object to use with the Attri2Vec model
            in the Keras method ``predict``.

        """
        node_ilocs = self.graph.node_ids_to_ilocs(node_ids.index)
        return NodeSequence(
            self.sample_features, self.batch_size, node_ilocs, shuffle=False
        )


class Node2VecNodeGenerator(BatchedNodeGenerator):
    """
    A data generator for node representation prediction with Node2Vec models.

    At minimum, supply the StellarGraph and the batch size.

    The supplied graph should be a StellarGraph object that is ready for
    machine learning. Currently the model does not require node features for
    nodes in the graph.

    Use the :meth:`flow` method supplying the nodes to get an object
    that can be used as a Keras data generator.

    Example::

        G_generator = Node2VecNodeGenerator(G, 50)
        data_gen = G_generator.flow(node_ids)

    .. seealso::

       Model using this generator: :class:`.Node2Vec`.

       An example using this generator (see the model for more): `unsupervised representation learning <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/keras-node2vec-embeddings.html>`_.

       Related functionality: :class:`.Node2VecLinkGenerator` for training, link prediction, and related tasks.

    Args:
        G (StellarGraph): The machine-learning ready graph.
        batch_size (int): Size of batch to return.
        name (str or None): Name of the generator (optional).
    """

    def __init__(self, G, batch_size, name=None):
        super().__init__(G, batch_size, use_node_features=False)
        self.name = name

    def sample_features(self, head_nodes, batch_num):
        """
        Get the ids of the head nodes, and return these as a list of feature
        arrays for the Node2Vec algorithm.

        Args:
            head_nodes: An iterable of head nodes to perform sampling on.

        Returns:
            A list of feature arrays, with each element being the id of each
            head node.
        """

        return np.array(head_nodes)

    def flow(self, node_ids):
        """
        Creates a generator/sequence object for node representation prediction
        with the supplied node ids. This should be used with a trained ``Node2Vec``
        model in order to transform node ids to node embeddings. For training,
        see ``Node2VecLinkGenerator`` instead.

        The node IDs are the nodes to inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset/all of the nodes in the graph.

        Args:
            node_ids: an iterable of node IDs.

        Returns:
            A NodeSequence object to use with the Node2Vec model
            in the Keras method ``predict``.
        """
        node_ilocs = self.graph.node_ids_to_ilocs(node_ids)
        return NodeSequence(
            self.sample_features, self.batch_size, node_ilocs, shuffle=False
        )

    def flow_from_dataframe(self, node_ids):
        """
        Creates a generator/sequence object for node representation prediction
        by using the index of the supplied dataframe as the node ids.

        Args:
            node_ids: a Pandas DataFrame of node_ids.

        Returns:
            A NodeSequence object to use with the Node2Vec model
            in the Keras method ``predict``.
        """
        node_ilocs = self.graph.node_ids_to_ilocs(node_ids.index)
        return NodeSequence(
            self.sample_features, self.batch_size, node_ilocs, shuffle=False
        )
