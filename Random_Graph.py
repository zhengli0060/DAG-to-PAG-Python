'''
Source code from https://github.com/francescomontagna/causally/tree/main
Author: Francescomontagna
@inproceedings{montagna2023_assumptions,
    author = {Montagna, Francesco and Mastakouri, Atalanti and Eulig, Elias and Noceti, Nicoletta and Rosasco, Lorenzo and Janzing, Dominik and Aragam, Bryon and Locatello, Francesco},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {Assumption violations in causal discovery and the robustness of score matching},
    url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/93ed74938a54a73b5e4c52bbaf42ca8e-Paper-Conference.pdf},
    year = {2023}
} 
'''

'''
Rewritten by: Zheng Li
Date: 2025-04-08
'''

import random
import igraph as ig
import numpy as np
import networkx as nx
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
from numpy.core.multiarray import array as array




# ********************** #
#       Utilities        #
# ********************** #
def acyclic_orientation(A) -> np.array:
    """Convert an undirected graph to a directed acyclic graph (DAG)."""
    return np.triu(A, k=1)


def ig_to_adjmat(G: ig.Graph) -> np.array:
    return np.array(G.get_adjacency().data)


def max_edges_in_dag(num_nodes: int) -> int:
    """Compute the maximum number of edges allowed for a direcetd acyclic graph:

    The max number of edges is compute as `self.num_nodes*(self.num_nodes-1)/2`
    """
    return int(num_nodes * (num_nodes - 1) / 2)

def num_errors(order, adj) -> int:
    """Compute the number of errors in the ordering."""
    err = 0
    for i in range(len(order)):
        err += adj[order[i + 1 :], order[i]].sum()
    # print(f"Number of errors in the ordering: {err}")
    return err

class GraphGenerator(metaclass=ABCMeta):
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    @abstractmethod
    def get_random_graph(self) -> np.array:
        """Sample a random directed acyclic graph (DAG).

        Returns
        -------
        A: np.array
            Adjacency matrix representation of the output DAG.
            The presence of a directed edge from node ``i`` to node ``j``
            is denoted by ``A[i, j] = 1``. Absence of the edge is denote by
            ``A[i, j] = 0``.
        """
        raise NotImplementedError()

    def _manual_seed(self, seed: int) -> None:
        """Set manual seed for deterministic graph generation. If None, seed is not set."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _make_random_order(self, A: np.array) -> np.array:
        """Randomly permute nodes of A to avoid trivial ordering."""
        n_nodes = A.shape[0]
        permutation = np.random.permutation(range(n_nodes))
        A = A[permutation, :]
        A = A[:, permutation]
        return A
    

# ***************************** #
#  Erdos-Rényi Graphs Generator #
# ***************************** #


class ErdosRenyi(GraphGenerator):
    """
    Generator of Erdos-Renyi directed acyclic graphs.

    This class is a wrapper of the Erdos-Renyi graph sampler of the ``igraph`` Python packege.

    Parameters
    ----------
    num_nodes: int
        Number of nodes.
    expected_degree: int, default None
        Expected degree of each node.
        The value provided must be greater or equal than 1.
    p_edge: float, default None
        Probability of edge between each pair of nodes.
        Accepted values are in the range (0.1, 1]
    min_num_edges: int, default 1
        The minimum number of edges required in the graph.
        If 0, allows for empty graphs. The maximum value allowed
        is ``num_nodes * (num_nodes - 1) / 2``, corresponding to a DAG
        with all nodes connected.
    def_topological_order: bool, default True
        If True, the generated graph will be returned in topological order.
        If False, the graph will be returned not in topological order.
    def_dataframe: bool, default False
        If True, the generated graph will be returned as a pandas dataframe.
        If False, the graph will be returned as a numpy array.
    seed: int, default None
        Random seed for reproducibility.
        If None, the random seed is not set.
        
    Notes
    -----
    One and only one parameter between `expected_degree` and `p_edge` must be explicitly provided.
    """

    def __init__(
        self,
        num_nodes: int,
        expected_degree: int = None,
        p_edge: float = None,
        min_num_edges: int = 1,
        def_topological_order: bool = True,
        def_dataframe: bool = False,
        seed: int = None
    ):
        if expected_degree is not None and p_edge is not None:
            raise ValueError(
                "Only one parameter between 'p_edge' and 'expected_degree' can be"
                f" provided. Got instead expected_degree={expected_degree}"
                f"and p_edge={p_edge}."
            )
        if expected_degree is None and p_edge is None:
            raise ValueError(
                "Please provide a value for one and only one argument between"
                " 'expected_degree' and 'p_edge'."
            )
        if expected_degree is not None and expected_degree == 0:
            raise ValueError(
                "expected value of 'expected_degree' is at least 1. Got 0 instead"
            )
        if p_edge is not None and p_edge < 0.1:
            raise ValueError(
                "expected value of 'p_edge' is at least 0.0001." f" Got {p_edge} instead"
            )
        if min_num_edges < 0:
            raise ValueError(
                "Minimum number of edges must be larger or equals to 0."
                + f" Got instead {min_num_edges}."
            )


        super().__init__(num_nodes)
        self.expected_degree = expected_degree
        self.p_edge = p_edge
        self.min_num_edges = min_num_edges
        self.def_topological_order = def_topological_order
        self.def_dataframe = def_dataframe
        self._manual_seed(seed)

    def get_random_graph(self, def_dataframe: bool=None) -> np.array:

        if def_dataframe is None:
            def_dataframe = self.def_dataframe

        A = -np.ones((self.num_nodes, self.num_nodes))

        """"
        - ``Erdos_Renyi(n, p)`` will generate a graph from the so-called :math:`G(n,p)` model where each edge between any two pair of nodes has an independent probability ``p`` of existing.
        - ``Erdos_Renyi(n, m)`` will pick a graph uniformly at random out of all graphs with ``n`` nodes and ``m`` edges. This is referred to as the :math:`G(n,m)` model.
        """
        # Ensure at least self.min_num_edges edges (one edge if the graph is bivariate)
        while np.sum(A) < min(self.min_num_edges, max_edges_in_dag(self.num_nodes)):
            if self.p_edge is not None:
                undirected_graph = ig.Graph.Erdos_Renyi(n=self.num_nodes, p=self.p_edge)  
            elif self.expected_degree is not None:
                undirected_graph = ig.Graph.Erdos_Renyi(
                    n=self.num_nodes, m=int(self.expected_degree * self.num_nodes * 0.5)
                )
            undirected_adjacency = ig_to_adjmat(undirected_graph)
            A = acyclic_orientation(undirected_adjacency)

        # Permute to avoid trivial ordering
        A = self._make_random_order(A)

        assert nx.is_directed_acyclic_graph(
            nx.from_numpy_array(A, create_using=nx.DiGraph)
        ), "The generated random graph is not acyclic! No topological order can be defined"

        if self.def_topological_order:
            # Get topological order
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
            topological_order = list(nx.topological_sort(G))
            A = A[topological_order, :][:, topological_order]

        if def_dataframe:
            # Convert to pandas dataframe
            A = pd.DataFrame(A, index=[f'V{i+1}' for i in range(self.num_nodes)], columns=[f'V{i+1}' for i in range(self.num_nodes)])

        return A
    


if __name__ == "__main__":
    # Example usage
    num_nodes = 5
    expected_degree = 2


    er_graph_gen = ErdosRenyi(num_nodes, expected_degree, def_dataframe=True, seed=12)
    random_graph = er_graph_gen.get_random_graph()
    print("Random Graph Adjacency Matrix:")
    print(random_graph)

   