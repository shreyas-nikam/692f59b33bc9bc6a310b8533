import pandas as pd
import networkx as nx
import random
import streamlit as st

# AI-BOM Risk Navigator Functions


def generate_ai_bom_dataset(num_components: int, num_dependencies: int) -> tuple[pd.DataFrame, list[tuple]]:
    """Generates a synthetic AI-BOM dataset and a list of dependencies."""
    component_types = ['Data', 'Model', 'Library', 'Hardware']
    origins = ['Internal', 'Third-Party Vendor A',
               'Third-Party Vendor B', 'Open Source Community']
    licensing_info = ['Open Source', 'Proprietary', 'MIT', 'GPLv3']

    components_data = []
    for i in range(num_components):
        component_id = f'Comp_{i:03d}'
        component_name = f'Component {i+1}'
        component_type = random.choice(component_types)
        origin = random.choice(origins)
        version = f'{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}'
        vulnerabilities_score = round(random.uniform(
            0, 10), 2)  # Simplified CVSS-like score
        license_info = random.choice(licensing_info)
        components_data.append([
            component_id, component_name, component_type, origin,
            version, vulnerabilities_score, license_info
        ])

    components_df = pd.DataFrame(components_data, columns=[
        'Component_ID', 'Component_Name', 'Component_Type', 'Origin',
        'Version', 'Known_Vulnerabilities_Score', 'Licensing_Info'
    ])

    dependencies = []
    possible_edges = num_components * (num_components - 1)
    if num_dependencies > possible_edges:
        num_dependencies = possible_edges
        st.warning(
            f"Warning: num_dependencies reduced to {num_dependencies} as it exceeded possible edges.")

    component_ids = components_df['Component_ID'].tolist()

    # Generate dependencies, ensuring no self-loops and minimizing duplicate edges
    for _ in range(num_dependencies):
        source, target = random.sample(component_ids, 2)
        if (source, target) not in dependencies:
            dependencies.append((source, target))

    return components_df, dependencies


def create_ai_system_graph(components_df: pd.DataFrame, dependencies: list[tuple]) -> nx.Graph:
    """Constructs a networkx graph object representing the AI system."""
    graph = nx.DiGraph()  # Directed Graph

    # Add nodes with attributes
    for _, row in components_df.iterrows():
        node_id = row['Component_ID']
        # Convert Series to dict for node attributes
        attributes = row.drop('Component_ID').to_dict()
        graph.add_node(node_id, **attributes)

    # Add edges
    graph.add_edges_from(dependencies)

    return graph


def calculate_component_risk_aibom(component_data: pd.Series) -> float:
    """Computes a risk score for an individual component based on its attributes."""
    risk_score = component_data['Known_Vulnerabilities_Score']

    # Add penalty for third-party origins
    if component_data['Origin'] in ['Third-Party Vendor A', 'Third-Party Vendor B']:
        risk_score += 2.0
    elif component_data['Origin'] == 'Open Source Community':
        risk_score += 1.0  # Slightly less penalty than commercial third-party

    # Ensure score is within a reasonable range (e.g., 0-15)
    risk_score = max(0.0, min(risk_score, 15.0))
    return round(risk_score, 2)


def get_component_risk_profile(components_df: pd.DataFrame, component_id: str) -> str:
    """Generates a human-readable textual risk profile for a specified component."""
    component_data = components_df[components_df['Component_ID']
                                   == component_id].iloc[0]

    profile = f"""
Component Risk Profile for: {component_data['Component_Name']} ({component_data['Component_ID']})
-----------------------------------------------------------------------
Component Type: {component_data['Component_Type']}
Origin: {component_data['Origin']}
Version: {component_data['Version']}
Known Vulnerabilities Score: {component_data['Known_Vulnerabilities_Score']:.2f}
Licensing Info: {component_data['Licensing_Info']}
Calculated Component Risk Score: {component_data['Component_Risk_Score']:.2f}
-----------------------------------------------------------------------
"""
    return profile


def aggregate_overall_ai_system_risk(graph: nx.Graph) -> float:
    """Calculates the overall risk for the entire AI system based on individual component risks and interdependencies."""
    total_weighted_risk = 0.0
    max_possible_weighted_risk = 0.0

    for node_id in graph.nodes():
        component_risk_score = graph.nodes[node_id].get(
            'Component_Risk_Score', 0.0)
        out_degree = graph.out_degree(node_id)

        weighted_risk = component_risk_score * (1 + out_degree)
        total_weighted_risk += weighted_risk

        max_component_risk_possible = 15.0  # Defined max from calculate_component_risk
        max_out_degree_possible = len(graph.nodes()) - 1
        max_possible_weighted_risk += max_component_risk_possible * \
            (1 + max_out_degree_possible)

    if max_possible_weighted_risk == 0:
        return 0.0

    normalized_overall_risk = (
        total_weighted_risk / max_possible_weighted_risk) * 100
    return round(normalized_overall_risk, 2)


def simulate_vulnerability_propagation(graph: nx.Graph, vulnerable_component_id: str, base_impact_score: float) -> nx.Graph:
    """Simulates a vulnerability in a specified component, propagates its impact, and updates risk scores."""
    simulated_graph = graph.copy()  # Create a deep copy to avoid modifying the original graph

    # 1. Directly update the vulnerable component
    if vulnerable_component_id in simulated_graph.nodes:
        # Get current component data to recalculate risk
        component_data_series = pd.Series(
            simulated_graph.nodes[vulnerable_component_id])

        # Artificially increase Known_Vulnerabilities_Score
        component_data_series['Known_Vulnerabilities_Score'] = min(
            component_data_series['Known_Vulnerabilities_Score'] + base_impact_score, 15.0)

        # Recalculate component risk using the updated vulnerability score
        updated_risk_score = calculate_component_risk_aibom(
            component_data_series)  # Reuse the risk calculation logic
        simulated_graph.nodes[vulnerable_component_id]['Component_Risk_Score'] = updated_risk_score
        # Update the vuln score in graph too
        simulated_graph.nodes[vulnerable_component_id]['Known_Vulnerabilities_Score'] = component_data_series['Known_Vulnerabilities_Score']
    else:
        st.warning(
            f"Vulnerable component '{vulnerable_component_id}' not found in the graph.")
        return simulated_graph  # Return original graph if component not found

    # 2. Propagate attenuated impact to direct downstream dependencies
    for successor in simulated_graph.successors(vulnerable_component_id):
        if successor in simulated_graph.nodes:
            current_risk = simulated_graph.nodes[successor].get(
                'Component_Risk_Score', 0.0)
            propagated_impact = base_impact_score * 0.5
            simulated_graph.nodes[successor]['Component_Risk_Score'] = min(
                current_risk + propagated_impact, 15.0)

            # Also slightly increase known vulnerabilities score for affected components for consistency
            current_vuln_score = simulated_graph.nodes[successor].get(
                'Known_Vulnerabilities_Score', 0.0)
            simulated_graph.nodes[successor]['Known_Vulnerabilities_Score'] = min(
                current_vuln_score + (base_impact_score * 0.2), 15.0)

            # 3. Propagate further attenuated impact to indirect downstream dependencies (dependencies of dependencies)
            for indirect_successor in simulated_graph.successors(successor):
                if indirect_successor in simulated_graph.nodes and indirect_successor != vulnerable_component_id:
                    current_risk_indirect = simulated_graph.nodes[indirect_successor].get(
                        'Component_Risk_Score', 0.0)
                    propagated_impact_indirect = base_impact_score * 0.2
                    simulated_graph.nodes[indirect_successor]['Component_Risk_Score'] = min(
                        current_risk_indirect + propagated_impact_indirect, 15.0)

                    current_vuln_score_indirect = simulated_graph.nodes[indirect_successor].get(
                        'Known_Vulnerabilities_Score', 0.0)
                    simulated_graph.nodes[indirect_successor]['Known_Vulnerabilities_Score'] = min(
                        current_vuln_score_indirect + (base_impact_score * 0.1), 15.0)

    return simulated_graph
