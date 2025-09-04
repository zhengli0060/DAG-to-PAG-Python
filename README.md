# DAG to PAG Python 🚀

A Python tool for converting Directed Acyclic Graphs (DAGs) into Partial Ancestral Graphs (PAGs) using the [causal-learn](https://github.com/cmu-phil/causal-learn) package.

## Features of `DAG_to_PAG.py` ✨
- 🔄 Converts a DAG adjacency matrix (NumPy array or Pandas DataFrame) into a PAG.
- 🧩 Supports specifying latent variables and selection bias nodes.
- ✅ Ensures compatibility with causal-learn's graph representation.

> Note: When the number of nodes is large, dag2pag runs slowly.

- For cases without selection bias, [FastDAG2PAG-python](https://github.com/zhengli0060/FastDAG2PAG-python) is recommended because it’s significantly faster.