from Random_Graph import ErdosRenyi
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Dag import Dag
from causallearn.utils.DAG2PAG import dag2pag
from causallearn.utils.GraphUtils import GraphUtils
import pandas as pd
import numpy as np
from typing import Union, List

def _create_Dag_nodes(adj_matrix: Union[pd.DataFrame, np.array]) -> List[GraphNode]:
    """
    Create a list of GraphNode objects from a list of node names.
    Parameters
    ----------
    nodes : list of node names

    Returns
    -------
    List of GraphNode objects
    """
    if isinstance(adj_matrix, pd.DataFrame):
        nodes = adj_matrix.index.tolist()
    elif isinstance(adj_matrix, np.ndarray):
        nodes = [f"V{i+1}" for i in range(adj_matrix.shape[0])]
    else:
        raise ValueError("adj_matrix must be a Pandas DataFrame or NumPy array.")
    
    return [GraphNode(name) for name in nodes], nodes  # Return both GraphNode objects and node names

def _create_Dag(adj_matrix: Union[pd.DataFrame, np.array], dag_nodes: List[GraphNode]) -> Dag:
    """
    Create a DAG from an adjacency matrix and a list of node names.
    Parameters
    ----------
    adj_matrix : Direct Acyclic Graph as a DataFrame or NumPy array
    node_names : list of node names

    Returns
    -------
    Dag : Directed Acyclic Graph as a Dag object
    """
    if isinstance(adj_matrix, pd.DataFrame):
        adj_matrix = adj_matrix.values  # Convert DataFrame to NumPy array if needed  

    # Create Dag  
    dag = Dag(dag_nodes)

    # Adds directed edges --> to the graph.
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1: # If there is an directed  edge from i to j
                dag.add_directed_edge(dag_nodes[i], dag_nodes[j])

    return dag



def DAG_to_PAG(adj_matrix: Union[pd.DataFrame, np.array], latent_nodes: list[Union[str, int]] = [], selection_bias_nodes: list[Union[str, int]] = []) -> Union[pd.DataFrame, np.array]:
    """
    Convert a DAG to its corresponding PAG based on 'causal-learn' Package.
    Parameters
    ----------
    adj_matrix : Direct Acyclic Graph as a DataFrame or NumPy array
    latent_nodes : list of latent nodes. [] means there is no latent variable
    selection_bias_nodes : list of selection bias nodes. [] means there is no selection variable

    Returns
    -------
    PAG : Partial Ancestral Graph as a DataFrame or NumPy array
    """
     
    # Check the adjacency matrix, latent nodes, and selection bias nodes
    if isinstance(adj_matrix, pd.DataFrame):
        if not ((adj_matrix.values == 0) | (adj_matrix.values == 1)).all():
            raise ValueError("All elements in the DataFrame must be 0 or 1.")
        if not all(isinstance(node, str) for node in adj_matrix.columns):
            raise ValueError("All column names in the DataFrame must be strings.")
        if not all(isinstance(node, str) for node in latent_nodes):
            raise ValueError("latent_nodes must be a string when adj_matrix is a DataFrame.")
        if not all(isinstance(node, str) for node in selection_bias_nodes):
            raise ValueError("selection_bias_nodes must be a string when adj_matrix is a DataFrame.")
    elif isinstance(adj_matrix, np.ndarray):
        if not ((adj_matrix == 0) | (adj_matrix == 1)).all():
            raise ValueError("All elements in the NumPy array must be 0 or 1.")
        if not all(isinstance(node, int) for node in latent_nodes):
            raise ValueError("latent_nodes must be an int when adj_matrix is a NumPy array.")
        if not all(isinstance(node, int) for node in selection_bias_nodes):
            raise ValueError("selection_bias_nodes must be an int when adj_matrix is a NumPy array.")
    else:
        raise ValueError("adj_matrix must be a Pandas DataFrame or NumPy array.")


    

    if isinstance(adj_matrix, pd.DataFrame):
        if any(node not in adj_matrix.columns for node in latent_nodes):
            raise ValueError("Some latent_nodes are not found in the DataFrame columns.")
        if any(node not in adj_matrix.columns for node in selection_bias_nodes):
            raise ValueError("Some selection_bias_nodes are not found in the DataFrame columns.")
        latent_nodes = [
            adj_matrix.columns.get_loc(node) for node in latent_nodes
        ]
        selection_bias_nodes = [
            adj_matrix.columns.get_loc(node) for node in selection_bias_nodes
        ]
    elif isinstance(adj_matrix, np.ndarray):
        if any(node >= adj_matrix.shape[0] or node < 0 for node in latent_nodes):
            raise ValueError("Some latent_nodes indices are out of range for the NumPy array.")
        if any(node >= adj_matrix.shape[0] or node < 0 for node in selection_bias_nodes):
            raise ValueError("Some selection_bias_nodes indices are out of range for the NumPy array.")
        
    dag_nodes, all_nodes = _create_Dag_nodes(adj_matrix)  
    dag = _create_Dag(adj_matrix, dag_nodes)
    
    islatent = [dag_nodes[i] for i in latent_nodes]  # Convert indices to GraphNode objects
    isselection = [dag_nodes[i] for i in selection_bias_nodes]  # Convert indices to GraphNode objects

    pag = dag2pag(dag, islatent, isselection)


    """
    test for draw pag
    """
    # # graphviz_pag = GraphUtils.to_pgv(pag)
    # # graphviz_pag.draw("pag.png", prog='dot', format='png')
    # # graphviz_pag.write("pag.dot")
    # # & "C:\Program Files\Graphviz\bin\dot.exe" -Tpng "pag.dot" -o "pag.png"


    """
    test for visible edge
    """
    # edge = pag.get_edge(dag_nodes[-2], dag_nodes[-1])
    # isvisible = pag.def_visible(edge)
    # print(f"Edge {dag_nodes[-2].get_name()} -> {dag_nodes[-1].get_name()} is visible: {isvisible}")


    #######################################################
    # In causal-learn, the graph is represented as a numpy array.
    # The graph[i, j] = 1 indicates that the mark near i on the edge between node i and node j is arrow.  
    # Adds a directed edge --> to the graph.
    # def add_directed_edge(self, node1: Node, node2: Node):
    #     i = self.node_map[node1]
    #     j = self.node_map[node2]
    #     self.graph[j, i] = 1
    #     self.graph[i, j] = -1
        # TAIL = -1
        # NULL = 0
        # ARROW = 1
        # CIRCLE = 2
    #######################################################
    pag_graph_node_name = [node.get_name() for node in pag.node_map.keys()]
    pag_graph_df = pd.DataFrame(pag.graph.T, columns=pag_graph_node_name, index=pag_graph_node_name) 
    observed_nodes = [all_nodes[i] for i in range(len(all_nodes)) if (i not in latent_nodes and i not in selection_bias_nodes)]
    pag_graph_df = pag_graph_df.loc[observed_nodes, observed_nodes]
    
    return pag_graph_df if isinstance(adj_matrix, pd.DataFrame) else pag_graph_df.to_numpy()  # Return as DataFrame or NumPy array based on input type



if __name__ == "__main__":

    # Example usage
    num_nodes = 5
    expected_degree = 2

    ER_graph_gen = ErdosRenyi(num_nodes, expected_degree, def_dataframe=False, seed=12)


    adj_matrix = ER_graph_gen.get_random_graph()
    print("Pretreatment adjacency matrix:\n", adj_matrix)
    pag = DAG_to_PAG(adj_matrix, latent_nodes=[0])
    print("PAG adjacency matrix:\n", pag)

    ER_graph_gen = ErdosRenyi(num_nodes, expected_degree, def_dataframe=True, seed=12)
    adj_matrix = ER_graph_gen.get_random_graph()
    print("Pretreatment adjacency matrix:\n", adj_matrix)
    pag = DAG_to_PAG(adj_matrix, latent_nodes=['V1'])
    print("PAG adjacency matrix:\n", pag)

