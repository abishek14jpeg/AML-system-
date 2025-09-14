# backend/graph_analysis.py

import pandas as pd
import networkx as nx
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_data.csv')
df = pd.read_csv(DATA_PATH)

def analyze_account_graph(account_id):
    """
    Builds a transaction graph for a specific account and detects cycles.
    """
    # Filter for transactions involving the target account
    account_df = df[(df['sender_account_id'] == account_id) | (df['receiver_account_id'] == account_id)]

    if account_df.empty:
        return {"nodes": [], "edges": [], "patterns": {"circular_transfers": []}}

    # Create a directed graph
    G = nx.from_pandas_edgelist(
        account_df,
        source='sender_account_id',
        target='receiver_account_id',
        edge_attr='amount',
        create_using=nx.DiGraph()
    )

    # Detect circular transfers (cycles in the graph)
    cycles = list(nx.simple_cycles(G))

    # Format graph data for visualization
    nodes = [{"id": str(node)} for node in G.nodes()]
    edges = [{"source": str(u), "target": str(v), "label": f"${w['amount']}"} for u, v, w in G.edges(data=True)]

    return {
        "nodes": nodes,
        "edges": edges,
        "patterns": {
            "circular_transfers": cycles
        }
    }