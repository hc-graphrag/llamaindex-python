import networkx as nx
from cdlib import algorithms

# Adapted from ms-graphrag/graphrag/index/utils/stable_lcc.py
def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Returns the largest connected component of a graph."""
    if not graph:
        return nx.Graph()
    
    components = list(nx.connected_components(graph))
    if not components:
        return nx.Graph()
    
    largest_component_nodes = max(components, key=len)
    return graph.subgraph(largest_component_nodes).copy()

# Adapted from ms-graphrag/graphrag/index/operations/cluster_graph.py
def _compute_leiden_communities(
    graph: nx.Graph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Return Leiden root communities and their hierarchy mapping."""
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    if len(graph.nodes()) == 0:
        return {}, {}

    # Create a mapping from original node names to integers
    original_nodes = list(graph.nodes())
    node_to_int = {node: i for i, node in enumerate(original_nodes)}
    int_to_node = {i: node for i, node in enumerate(original_nodes)}
    
    # Create a new graph with integer node names
    int_graph = nx.Graph()
    for node in original_nodes:
        int_graph.add_node(node_to_int[node])
    
    for edge in graph.edges():
        int_graph.add_edge(node_to_int[edge[0]], node_to_int[edge[1]])

    # Use cdlib's leiden algorithm on the integer graph
    if seed is not None:
        import random
        random.seed(seed)
    communities = algorithms.leiden(int_graph)
    
    # Create results mapping using original node names
    results: dict[int, dict[str, int]] = {0: {}}
    hierarchy: dict[int, int] = {}
    
    for i, community in enumerate(communities.communities):
        hierarchy[i] = -1  # cdlib's leiden doesn't provide hierarchical structure by default
        for int_node in community:
            original_node = int_to_node[int_node]
            results[0][str(original_node)] = i
    
    return results, hierarchy

# Adapted from ms-graphrag/graphrag/index/operations/cluster_graph.py
def cluster_graph(
    graph: nx.Graph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> list[tuple[int, int, int, list[str]]]: # Communities type
    """Apply a hierarchical clustering algorithm to a graph."""
    if len(graph.nodes) == 0:
        print("Graph has no nodes for clustering.")
        return []

    node_id_to_community_map, parent_mapping = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=seed,
    )

    levels = sorted(node_id_to_community_map.keys())

    clusters: dict[int, dict[int, list[str]]] = {}
    for level in levels:
        result = {}
        clusters[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = raw_community_id
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)

    results: list[tuple[int, int, int, list[str]]] = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, parent_mapping.get(cluster_id, -1), nodes))
    return results
