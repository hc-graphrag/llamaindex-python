import networkx as nx
from graspologic.partition import hierarchical_leiden

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
    """Return Leiden hierarchical communities and their hierarchy mapping."""
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

    # Use graspologic's hierarchical leiden algorithm
    hierarchical_clusters = hierarchical_leiden(
        int_graph,
        max_cluster_size=max_cluster_size,
        resolution=1.0,
        randomness=0.001,
        random_seed=seed
    )
    
    # Process hierarchical clustering results
    results: dict[int, dict[str, int]] = {}
    hierarchy: dict[int, int] = {}
    
    # Process each hierarchical cluster assignment
    for hcluster in hierarchical_clusters:
        cluster_level = hcluster.level
        cluster_id = hcluster.cluster
        int_node = hcluster.node
        parent_cluster_id = hcluster.parent_cluster
        
        # Initialize level in results if not exists
        if cluster_level not in results:
            results[cluster_level] = {}
        
        # Map node back to original name and assign to cluster
        if int_node in int_to_node:
            original_node = int_to_node[int_node]
            results[cluster_level][str(original_node)] = cluster_id
        
        # Set hierarchy mapping (parent cluster)
        hierarchy[cluster_id] = parent_cluster_id if parent_cluster_id is not None else -1
    
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
