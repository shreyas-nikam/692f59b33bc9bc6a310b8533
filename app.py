import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import io

# Configure plot styles for better readability (will be applied once at app start)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100

def generate_ai_bom_dataset(num_components: int, num_dependencies: int) -> tuple[pd.DataFrame, list[tuple]]:
    """Generates a synthetic AI-BOM dataset and a list of dependencies."""
    component_types = ['Data', 'Model', 'Library', 'Hardware']
    origins = ['Internal', 'Third-Party Vendor A', 'Third-Party Vendor B', 'Open Source Community']
    licensing_info = ['Open Source', 'Proprietary', 'MIT', 'GPLv3']

    components_data = []
    for i in range(num_components):
        component_id = f'Comp_{i:03d}'
        component_name = f'Component {i+1}'
        component_type = random.choice(component_types)
        origin = random.choice(origins)
        version = f'{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}'
        vulnerabilities_score = round(random.uniform(0, 10), 2) # Simplified CVSS-like score
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
        st.warning(f"Warning: num_dependencies reduced to {num_dependencies} as it exceeded possible edges.")

    component_ids = components_df['Component_ID'].tolist()

    # Generate dependencies, ensuring no self-loops and minimizing duplicate edges
    for _ in range(num_dependencies):
        source, target = random.sample(component_ids, 2)
        if (source, target) not in dependencies:
            dependencies.append((source, target))

    return components_df, dependencies

def create_ai_system_graph(components_df: pd.DataFrame, dependencies: list[tuple]) -> nx.Graph:
    """Constructs a networkx graph object representing the AI system."""
    graph = nx.DiGraph() # Directed Graph

    # Add nodes with attributes
    for _, row in components_df.iterrows():
        node_id = row['Component_ID']
        # Convert Series to dict for node attributes
        attributes = row.drop('Component_ID').to_dict()
        graph.add_node(node_id, **attributes)

    # Add edges
    graph.add_edges_from(dependencies)

    return graph

def calculate_component_risk(component_data: pd.Series) -> float:
    """Computes a risk score for an individual component based on its attributes."""
    risk_score = component_data['Known_Vulnerabilities_Score']

    # Add penalty for third-party origins
    if component_data['Origin'] in ['Third-Party Vendor A', 'Third-Party Vendor B']:
        risk_score += 2.0
    elif component_data['Origin'] == 'Open Source Community':
        risk_score += 1.0 # Slightly less penalty than commercial third-party

    # Ensure score is within a reasonable range (e.g., 0-15)
    risk_score = max(0.0, min(risk_score, 15.0))
    return round(risk_score, 2)

def get_component_risk_profile(components_df: pd.DataFrame, component_id: str) -> str:
    """Generates a human-readable textual risk profile for a specified component."""
    component_data = components_df[components_df['Component_ID'] == component_id].iloc[0]

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
        component_risk_score = graph.nodes[node_id].get('Component_Risk_Score', 0.0)
        out_degree = graph.out_degree(node_id)

        weighted_risk = component_risk_score * (1 + out_degree)
        total_weighted_risk += weighted_risk

        max_component_risk_possible = 15.0 # Defined max from calculate_component_risk
        max_out_degree_possible = len(graph.nodes()) - 1
        max_possible_weighted_risk += max_component_risk_possible * (1 + max_out_degree_possible)

    if max_possible_weighted_risk == 0:
        return 0.0

    normalized_overall_risk = (total_weighted_risk / max_possible_weighted_risk) * 100
    return round(normalized_overall_risk, 2)

def simulate_vulnerability_propagation(graph: nx.Graph, vulnerable_component_id: str, base_impact_score: float) -> nx.Graph:
    """Simulates a vulnerability in a specified component, propagates its impact, and updates risk scores."""
    simulated_graph = graph.copy() # Create a deep copy to avoid modifying the original graph

    # 1. Directly update the vulnerable component
    if vulnerable_component_id in simulated_graph.nodes:
        # Get current component data to recalculate risk
        component_data_series = pd.Series(simulated_graph.nodes[vulnerable_component_id])

        # Artificially increase Known_Vulnerabilities_Score
        component_data_series['Known_Vulnerabilities_Score'] = min(component_data_series['Known_Vulnerabilities_Score'] + base_impact_score, 15.0)

        # Recalculate component risk using the updated vulnerability score
        updated_risk_score = calculate_component_risk(component_data_series) # Reuse the risk calculation logic
        simulated_graph.nodes[vulnerable_component_id]['Component_Risk_Score'] = updated_risk_score
        simulated_graph.nodes[vulnerable_component_id]['Known_Vulnerabilities_Score'] = component_data_series['Known_Vulnerabilities_Score'] # Update the vuln score in graph too
    else:
        st.warning(f"Vulnerable component '{vulnerable_component_id}' not found in the graph.")
        return simulated_graph # Return original graph if component not found

    # 2. Propagate attenuated impact to direct downstream dependencies
    for successor in simulated_graph.successors(vulnerable_component_id):
        if successor in simulated_graph.nodes:
            current_risk = simulated_graph.nodes[successor].get('Component_Risk_Score', 0.0)
            propagated_impact = base_impact_score * 0.5
            simulated_graph.nodes[successor]['Component_Risk_Score'] = min(current_risk + propagated_impact, 15.0)

            # Also slightly increase known vulnerabilities score for affected components for consistency
            current_vuln_score = simulated_graph.nodes[successor].get('Known_Vulnerabilities_Score', 0.0)
            simulated_graph.nodes[successor]['Known_Vulnerabilities_Score'] = min(current_vuln_score + (base_impact_score * 0.2), 15.0)

            # 3. Propagate further attenuated impact to indirect downstream dependencies (dependencies of dependencies)
            for indirect_successor in simulated_graph.successors(successor):
                if indirect_successor in simulated_graph.nodes and indirect_successor != vulnerable_component_id:
                    current_risk_indirect = simulated_graph.nodes[indirect_successor].get('Component_Risk_Score', 0.0)
                    propagated_impact_indirect = base_impact_score * 0.2
                    simulated_graph.nodes[indirect_successor]['Component_Risk_Score'] = min(current_risk_indirect + propagated_impact_indirect, 15.0)

                    current_vuln_score_indirect = simulated_graph.nodes[indirect_successor].get('Known_Vulnerabilities_Score', 0.0)
                    simulated_graph.nodes[indirect_successor]['Known_Vulnerabilities_Score'] = min(current_vuln_score_indirect + (base_impact_score * 0.1), 15.0)

    return simulated_graph

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, the **AI-BOM Risk Navigator** is an interactive Streamlit application designed for Risk Managers to explore and assess risks within AI system supply chains. It provides a visual and data-driven approach to understanding component dependencies, identifying vulnerabilities, and simulating the cascading impact of compromised elements.

**Learning Goals:**
- Understand the concept and importance of an AI Bill of Materials (AI-BOM) for AI risk management.
- Learn how to generate and interpret a synthetic AI-BOM dataset.
- Visualize AI system dependencies and their structural integrity.
- Calculate and interpret individual component risk scores based on attributes like vulnerabilities and origin.
- Aggregate overall AI System Risk, accounting for component interdependencies.
- Simulate the propagation of a vulnerability throughout the AI system and visualize its impact.
- Assess specific risks related to data provenance and third-party components.
- Identify top and bottom risk components to prioritize mitigation efforts.
""")

# Initialize session state variables if they don't exist
if 'ai_bom_df' not in st.session_state:
    st.session_state.ai_bom_df = pd.DataFrame()
if 'ai_bom_dependencies' not in st.session_state:
    st.session_state.ai_bom_dependencies = []
if 'ai_system_graph' not in st.session_state:
    st.session_state.ai_system_graph = nx.DiGraph()
if 'simulated_risk_graph' not in st.session_state:
    st.session_state.simulated_risk_graph = nx.DiGraph()
if 'num_components_input' not in st.session_state:
    st.session_state.num_components_input = 15
if 'num_dependencies_input' not in st.session_state:
    st.session_state.num_dependencies_input = 20
if 'vulnerable_component_id_input' not in st.session_state:
    st.session_state.vulnerable_component_id_input = None
if 'base_impact_score_input' not in st.session_state:
    st.session_state.base_impact_score_input = 7.0
if 'num_top_bottom_n_input' not in st.session_state:
    st.session_state.num_top_bottom_n_input = 3


# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    st.subheader("1. Generate AI-BOM Dataset")
    st.session_state.num_components_input = st.slider("Number of Components", min_value=5, max_value=50, value=st.session_state.num_components_input, key='num_components_slider')
    st.session_state.num_dependencies_input = st.slider("Number of Dependencies", min_value=5, max_value=100, value=st.session_state.num_dependencies_input, key='num_dependencies_slider')
    if st.button("Generate AI-BOM", key='generate_button'):
        # Call generate_ai_bom_dataset and update session state
        st.session_state.ai_bom_df, st.session_state.ai_bom_dependencies = generate_ai_bom_dataset(
            st.session_state.num_components_input, 
            st.session_state.num_dependencies_input
        )
        # Re-calculate component risks and create initial graph
        st.session_state.ai_bom_df['Component_Risk_Score'] = st.session_state.ai_bom_df.apply(calculate_component_risk, axis=1)
        st.session_state.ai_system_graph = create_ai_system_graph(st.session_state.ai_bom_df, st.session_state.ai_bom_dependencies)
        st.session_state.simulated_risk_graph = st.session_state.ai_system_graph.copy() # Initialize simulated graph with current state
        st.success("AI-BOM Generated!")

    if not st.session_state.ai_bom_df.empty:
        st.subheader("2. Simulate Vulnerability")
        component_ids = st.session_state.ai_bom_df['Component_ID'].tolist()
        st.session_state.vulnerable_component_id_input = st.selectbox(
            "Select Vulnerable Component ID", 
            options=component_ids, 
            index=0 if component_ids else None, 
            key='vulnerable_component_select'
        )
        st.session_state.base_impact_score_input = st.slider(
            "Base Impact Score", min_value=0.0, max_value=10.0, step=0.5, 
            value=st.session_state.base_impact_score_input, key='base_impact_slider'
        )
        if st.button("Run Vulnerability Simulation", key='simulate_button'):
            if st.session_state.vulnerable_component_id_input:
                # Ensure the graph used for simulation is the current 'initial' graph state
                # to allow re-running simulation from a fresh state if AI-BOM was regenerated
                current_graph_for_simulation = create_ai_system_graph(st.session_state.ai_bom_df, st.session_state.ai_bom_dependencies)
                # Ensure risk scores are up to date on this current graph
                for index, row in st.session_state.ai_bom_df.iterrows():
                    current_graph_for_simulation.nodes[row['Component_ID']]['Component_Risk_Score'] = row['Component_Risk_Score']
                    current_graph_for_simulation.nodes[row['Component_ID']]['Known_Vulnerabilities_Score'] = row['Known_Vulnerabilities_Score']

                st.session_state.simulated_risk_graph = simulate_vulnerability_propagation(
                    current_graph_for_simulation, 
                    st.session_state.vulnerable_component_id_input, 
                    st.session_state.base_impact_score_input
                )
                st.success(f"Simulation complete for '{st.session_state.vulnerable_component_id_input}'!")
            else:
                st.warning("Please generate an AI-BOM first and select a vulnerable component.")


page = st.sidebar.selectbox(label="Navigation", options=[
    "Introduction", "AI-BOM Details", "Initial Dependencies", 
    "Component Risk Profile", "Vulnerability Impact", "Targeted Risk Analysis"
])

if page == "Introduction":
    from application_pages.page_1_introduction import main
    main()
elif page == "AI-BOM Details":
    from application_pages.page_2_ai_bom_details import main
    main()
elif page == "Initial Dependencies":
    from application_pages.page_3_initial_dependencies import main
    main()
elif page == "Component Risk Profile":
    from application_pages.page_4_component_risk_profile import main
    main()
elif page == "Vulnerability Impact":
    from application_pages.page_5_vulnerability_impact import main
    main()
elif page == "Targeted Risk Analysis":
    from application_pages.page_6_targeted_risk_analysis import main
    main()
