"""graph_operations (graspologic_native 版 hierarchical Leiden) のテスト。

Leiden は確率的アルゴリズムで、graspologic-native の版が変わると同一シードでも分割結果が
変わりうる (1.2.5 と 1.3.1 でモジュラリティは同等だが出力は不一致)。そのため特定版の出力を
固定せず、版に依存しない性質を検証する。
"""

import networkx as nx

from graphrag_anthropic_llamaindex.graph_operations import cluster_graph


def _modularity_at_level(graph, result, level):
    graph = nx.relabel_nodes(graph, str)
    parts = [set(map(str, nodes)) for lv, _, _, nodes in result if lv == level]
    covered = set().union(*parts)
    return nx.community.modularity(graph.subgraph(covered), parts)


def _leaf_sizes(result):
    parents = {parent for _, _, parent, _ in result}
    return [len(nodes) for _, cid, _, nodes in result if cid not in parents]


def test_same_seed_is_deterministic():
    graph = nx.planted_partition_graph(6, 25, 0.35, 0.01, seed=3)
    assert cluster_graph(graph, 10, True, 42) == cluster_graph(graph, 10, True, 42)


def test_result_shape_and_node_coverage_at_level0():
    graph = nx.karate_club_graph()
    result = cluster_graph(graph, 10, True, 42)

    for level, cluster_id, parent, nodes in result:
        assert isinstance(level, int) and isinstance(cluster_id, int) and isinstance(parent, int)
        assert nodes and all(isinstance(n, str) for n in nodes)

    level0 = [nodes for level, _, _, nodes in result if level == 0]
    flat = [n for nodes in level0 for n in nodes]
    assert sorted(flat) == sorted(str(n) for n in graph.nodes)
    assert len(flat) == len(set(flat)), "level 0 でノードが複数クラスタに重複している"
    assert all(parent == -1 for level, _, parent, _ in result if level == 0)


def test_community_quality_on_karate_club():
    result = cluster_graph(nx.karate_club_graph(), 10, True, 42)
    assert _modularity_at_level(nx.karate_club_graph(), result, 0) > 0.40


def test_hierarchy_splits_until_max_cluster_size():
    graph = nx.planted_partition_graph(6, 25, 0.35, 0.01, seed=3)
    result = cluster_graph(graph, 10, True, 42)

    assert max(level for level, *_ in result) >= 1
    assert max(_leaf_sizes(result)) <= 10
    assert sum(_leaf_sizes(result)) == graph.number_of_nodes()

    cluster_ids = {cid for _, cid, _, _ in result}
    assert all(parent == -1 or parent in cluster_ids for _, _, parent, _ in result)


def test_use_lcc_keeps_only_largest_component():
    graph = nx.disjoint_union(nx.karate_club_graph(), nx.path_graph(5))
    clustered = {n for _, _, _, nodes in cluster_graph(graph, 10, True, 42) for n in nodes}
    assert clustered == {str(n) for n in range(34)}


def test_isolated_nodes_are_not_clustered_without_lcc():
    graph = nx.disjoint_union(nx.karate_club_graph(), nx.path_graph(5))
    graph.add_nodes_from([900, 901])
    clustered = {n for _, _, _, nodes in cluster_graph(graph, 10, False, 42) for n in nodes}
    assert {"900", "901"}.isdisjoint(clustered)
    assert {str(n) for n in range(39)} <= clustered


def test_string_node_names_are_preserved():
    graph = nx.relabel_nodes(nx.connected_caveman_graph(4, 5), lambda i: f"エンティティ{i}")
    result = cluster_graph(graph, 5, True, 42)
    assert {n for _, _, _, nodes in result for n in nodes} == set(graph.nodes)


def test_empty_graph_returns_empty_list():
    assert cluster_graph(nx.Graph(), 10, True, 42) == []
