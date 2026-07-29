from agents.supervisor import build_graph


def test_graph_compiles_without_node_state_name_collisions() -> None:
    assert build_graph() is not None



def test_every_specialist_has_a_required_deliberation_round() -> None:
    graph = build_graph().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    specialties = (
        "radiology", "pathology", "surgery", "radiation",
        "medical_oncology", "supportive",
    )
    for specialty in specialties:
        assert (f"{specialty}_agent", "initial_aggregate_node") in edges
        assert ("initial_aggregate_node", f"{specialty}_deliberation_agent") in edges
        assert (f"{specialty}_deliberation_agent", "aggregate_node") in edges
    assert ("aggregate_node", "consensus_node") in edges
