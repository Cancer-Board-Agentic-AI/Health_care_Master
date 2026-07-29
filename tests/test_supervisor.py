from agents.supervisor import build_graph


def test_graph_compiles_without_node_state_name_collisions() -> None:
    assert build_graph() is not None
