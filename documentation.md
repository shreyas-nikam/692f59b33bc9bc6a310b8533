id: 692f59b33bc9bc6a310b8533_documentation
summary: AI Design and Deployment Lab 6 Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# AI-BOM Risk Navigator: A Comprehensive Guide for AI Supply Chain Risk Management

## 1. Introduction to AI Bill of Materials (AI-BOM) for AI Risk Management
Duration: 0:05
This codelab introduces the **AI-BOM Risk Navigator**, a Streamlit application designed for understanding, visualizing, and managing risks within AI system supply chains. As AI systems become increasingly complex, comprising various datasets, models, libraries, and hardware, a structured inventory of these components—an AI Bill of Materials (AI-BOM)—becomes indispensable.

Just as a Software Bill of Materials (SBOM) provides transparency for software components, an AI-BOM extends this concept to the unique elements of AI systems. This transparency is crucial for:
*   **Identifying vulnerabilities**: Pinpointing weaknesses in specific components.
*   **Assessing data provenance**: Understanding the origin and trustworthiness of training data.
*   **Managing legal and compliance risks**: Tracking licensing information and regulatory adherence.
*   **Understanding cascading impacts**: Predicting how a compromise in one component can affect the entire system.
*   **Proactive risk mitigation**: Enabling data-driven decisions for strengthening AI system security and resilience.

**Learning Goals:**
By the end of this codelab, you will be able to:
*   Understand the fundamental concepts of an AI Bill of Materials (AI-BOM) and its role in AI risk management.
*   Navigate and interact with the AI-BOM Risk Navigator Streamlit application.
*   Generate and interpret synthetic AI-BOM datasets and component attributes.
*   Visualize AI system dependencies using network graphs.
*   Calculate and explain individual component risk scores based on various attributes.
*   Aggregate and understand the overall AI system risk profile.
*   Simulate the propagation of vulnerabilities and visualize their cascading impact across the system.
*   Perform targeted risk analyses focusing on data provenance, third-party components, and critical risk areas.

<aside class="positive">
This codelab emphasizes a practical, hands-on approach to AI risk management, demonstrating how an AI-BOM can empower developers and risk managers to build more secure and trustworthy AI systems.
</aside>

## 2. Application Architecture Overview and Setup
Duration: 0:10

The AI-BOM Risk Navigator is a Streamlit application, providing an interactive web interface. It's structured into several Python files:
*   `app.py`: The main entry point of the Streamlit application. It handles the sidebar controls for generating data and running simulations, initializes session state, and orchestrates the navigation between different content pages. It also contains core utility functions for AI-BOM generation, graph creation, and risk calculations.
*   `application_pages/`: A directory containing separate Python files for each content page (`page_1_introduction.py`, `page_2_ai_bom_details.py`, etc.). Each of these files defines a `main()` function that renders the specific content for that page.

### Application Flow
The application follows a standard Streamlit pattern:
1.  **Sidebar Configuration**: Users interact with sliders and buttons in the sidebar to generate data or trigger simulations.
2.  **Session State Management**: All core data (AI-BOM DataFrame, dependency graph, simulated graph) is stored in Streamlit's `st.session_state`. This ensures data persists across reruns and page navigations without re-computation.
3.  **Page Navigation**: A `selectbox` in the sidebar allows users to navigate through different analysis pages.
4.  **Content Rendering**: Based on the selected page, the `main()` function of the corresponding page file is called, which then uses the data from `st.session_state` to render tables, graphs, and textual analysis.

### Visualizing the Application Architecture

```mermaid
graph TD
    User(User Interaction) --> StreamlitApp[Streamlit App (app.py)]

    subgraph StreamlitApp
        Sidebar[Sidebar (app.py)]
        MainContent[Main Content Area]
        SessionState(st.session_state)
    end

    Sidebar -- Configuration Inputs (Num Components, Dependencies, Vuln Simulation) --> SessionState
    Sidebar -- Triggers Generation/Simulation --> AppLogic[Core Application Logic (app.py functions)]

    AppLogic -- Updates AI-BOM Data, Graph, Risk Scores --> SessionState

    SessionState -- Data for Pages --> Page1[page_1_introduction.py]
    SessionState -- Data for Pages --> Page2[page_2_ai_bom_details.py]
    SessionState -- Data for Pages --> Page3[page_3_initial_dependencies.py]
    SessionState -- Data for Pages --> Page4[page_4_component_risk_profile.py]
    SessionState -- Data for Pages --> Page5[page_5_vulnerability_impact.py]
    SessionState -- Data for Pages --> Page6[page_6_targeted_risk_analysis.py]

    Page1 --> MainContent
    Page2 --> MainContent
    Page3 --> MainContent
    Page4 --> MainContent
    Page5 --> MainContent
    Page6 --> MainContent

    MainContent -- Displayed to User --> User
```

### Setup and Running the Application

To run this application locally, you would typically:
1.  Save the `app.py` file and the `application_pages` directory with its contents.
2.  Install the required Python libraries (e.g., `streamlit`, `pandas`, `numpy`, `networkx`, `matplotlib`, `seaborn`).
3.  Run `streamlit run app.py` from your terminal.

The `app.py` file sets up the basic Streamlit page configuration, initializes session state variables, and defines the core functions that are used across different pages.

```python
# app.py excerpt
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import io

# Configure plot styles for better readability
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, the **AI-BOM Risk Navigator** is an interactive Streamlit application...
""")

# Initialize session state variables
if 'ai_bom_df' not in st.session_state:
    st.session_state.ai_bom_df = pd.DataFrame()
# ... (other session state initializations) ...
```

<aside class="negative">
It is crucial to understand Streamlit's session state. Variables stored in `st.session_state` persist across reruns of the script and between different pages, enabling a consistent user experience. Forgetting to initialize or properly update session state can lead to unexpected behavior.
</aside>

## 3. Generating the AI-BOM Dataset
Duration: 0:08

The first interactive step in the application is to generate a synthetic AI Bill of Materials dataset. This dataset simulates the diverse components that constitute an AI system, along with their interdependencies. The `app.py` file contains the `generate_ai_bom_dataset` function responsible for this.

### `generate_ai_bom_dataset` Function

This function takes `num_components` and `num_dependencies` as input to create a DataFrame of components and a list of directed dependencies (edges).

```python
# app.py excerpt
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
```

### Interacting with the Sidebar Controls
In the Streamlit application, you can adjust the number of components and dependencies using sliders in the sidebar.

```python
# app.py excerpt (sidebar section)
with st.sidebar:
    st.header("Configuration")
    st.subheader("1. Generate AI-BOM Dataset")
    st.session_state.num_components_input = st.slider("Number of Components", min_value=5, max_value=50, value=st.session_state.num_components_input, key='num_components_slider')
    st.session_state.num_dependencies_input = st.slider("Number of Dependencies", min_value=5, max_value=100, value=st.session_state.num_dependencies_input, key='num_dependencies_slider')
    if st.button("Generate AI-BOM", key='generate_button'):
        st.session_state.ai_bom_df, st.session_state.ai_bom_dependencies = generate_ai_bom_dataset(
            st.session_state.num_components_input, 
            st.session_state.num_dependencies_input
        )
        # ... (further processing after generation) ...
        st.success("AI-BOM Generated!")
```

Clicking the "Generate AI-BOM" button will trigger the `generate_ai_bom_dataset` function, populate `st.session_state.ai_bom_df` and `st.session_state.ai_bom_dependencies`, and then proceed to calculate initial component risks and construct the initial dependency graph.

## 4. Understanding AI-BOM Component Attributes
Duration: 0:05

Once the AI-BOM is generated, the "AI-BOM Details" page (`application_pages/page_2_ai_bom_details.py`) displays the synthetic dataset and explains the significance of each attribute.

```python
# application_pages/page_2_ai_bom_details.py excerpt
def main():
    st.header("2. Generating a Synthetic AI-BOM Dataset")
    st.markdown("""
    To demonstrate the utility of an AI-BOM, we will generate a synthetic dataset...
    """)
    if not st.session_state.ai_bom_df.empty:
        st.subheader("AI-BOM DataFrame")
        st.dataframe(st.session_state.ai_bom_df)
        st.subheader("3. Understanding Component Attributes")
        st.markdown("""
        Each entry in the AI-BOM describes a component with several key attributes...
        *   **Component Name/ID**: A unique identifier for the component.
        *   **Component Type**: Categorizes the component (e.g., 'Data', 'Model', 'Library', 'Hardware'). Different types may have different risk profiles.
        *   **Origin/Provenance**: Where the component came from (e.g., 'Internal', 'Third-Party Vendor A', 'Open Source Community'). Crucial for assessing supply chain risks.
        *   **Version**: Specific version details, important for tracking known vulnerabilities.
        *   **Known Vulnerabilities (Score)**: A numerical representation of identified security weaknesses. We use a simplified score, conceptually related to CVSS.
        *   **Licensing Information**: Details about the license, which can imply legal or security risks.
        """)
        st.markdown("### AI-BOM DataFrame Information:")
        info_buffer = io.StringIO()
        st.session_state.ai_bom_df.info(buf=info_buffer)
        st.text(info_buffer.getvalue())

        st.markdown("### AI-BOM DataFrame Descriptive Statistics:")
        st.dataframe(st.session_state.ai_bom_df.describe())
    else:
        st.info("Please generate an AI-BOM using the sidebar controls to see the dataset and its attributes.")
```

The attributes are crucial for assessing individual component risks:
*   **`Component_ID`**: Unique identifier.
*   **`Component_Name`**: Human-readable name.
*   **`Component_Type`**: Classifies the component (e.g., 'Data', 'Model').
*   **`Origin`**: Source of the component (e.g., 'Internal', 'Third-Party Vendor A', 'Open Source Community'). This is vital for supply chain risk assessment.
*   **`Version`**: Version number, critical for tracking known vulnerabilities and compatibility.
*   **`Known_Vulnerabilities_Score`**: A synthetic score representing the severity of known security weaknesses (conceptually similar to CVSS).
*   **`Licensing_Info`**: Details about the component's license, which can have legal and security implications.
*   **`Component_Risk_Score`**: (Added later in `app.py`) The calculated risk score for each component, derived from its attributes.

## 5. Visualizing AI System Dependencies (Initial Graph)
Duration: 0:07

Understanding the structural relationships between components is paramount. The application uses `networkx` to create a directed graph, where components are nodes and dependencies are edges. This visualization helps risk managers quickly grasp the complexity and interconnections.

### `create_ai_system_graph` Function

This function, located in `app.py`, takes the DataFrame of components and the list of dependencies to construct a `networkx.DiGraph`.

```python
# app.py excerpt
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
```

### Displaying the Initial Graph
The `application_pages/page_3_initial_dependencies.py` file is responsible for rendering this graph using `matplotlib`. Nodes are colored by `Component_Type` to provide a quick visual overview.

```python
# application_pages/page_3_initial_dependencies.py excerpt
def main():
    st.header("4. Visualizing AI System Dependencies (Initial Graph)")
    st.markdown("""
    A graphical representation of the AI system, where components are nodes and dependencies are edges...
    """)
    if not st.session_state.ai_system_graph.empty():
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(st.session_state.ai_system_graph, seed=42)

        component_types = [nx.get_node_attributes(st.session_state.ai_system_graph, 'Component_Type')[node] for node in st.session_state.ai_system_graph.nodes()]
        unique_types = list(set(component_types))
        colors = plt.cm.get_cmap('tab10', len(unique_types))
        color_map = {ctype: colors(i) for i, ctype in enumerate(unique_types)}
        node_colors = [color_map[nx.get_node_attributes(st.session_state.ai_system_graph, 'Component_Type')[node]] for node in st.session_state.ai_system_graph.nodes()]

        nx.draw_networkx_nodes(st.session_state.ai_system_graph, pos, node_color=node_colors, node_size=3000, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(st.session_state.ai_system_graph, pos, edgelist=st.session_state.ai_system_graph.edges(), edge_color='gray', arrowsize=20, ax=ax)
        nx.draw_networkx_labels(st.session_state.ai_system_graph, pos, font_size=8, font_weight='bold', ax=ax)

        prompt_handles = [plt.Line2D([0], [0], marker='o', color='w', label=ctype,
                                   markerfacecolor=color_map[ctype], markersize=10)
                        for ctype in unique_types]
        ax.legend(handles=prompt_handles, title="Component Type")

        ax.set_title("Initial AI System Dependency Graph (AI-BOM)")
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("Generate an AI-BOM to visualize the initial dependency graph.")
```

## 6. Introduction to Vulnerability Scoring
Duration: 0:03

Before diving into component risk calculation, it's important to understand the concept of vulnerability scoring. The "Component Risk Profile" page (`application_pages/page_4_component_risk_profile.py`) begins by explaining this.

```python
# application_pages/page_4_component_risk_profile.py excerpt
def main():
    st.header("5. Introduction to Vulnerability Scoring")
    st.markdown("""
    Vulnerability scoring provides a standardized way to quantify the severity of security weaknesses. The Common Vulnerability Scoring System (CVSS) is a widely used open framework for communicating these characteristics and impacts. A simplified version of CVSS helps in prioritizing risks.

    The CVSS score is derived from various metrics, broadly categorized into base, temporal, and environmental metrics. For our simplified scenario, we can represent this as:
    $$ \text{CVSS Score} = g(\text{AttackVector}, \text{AttackComplexity}, \text{PrivilegesRequired}, \dots) $$
    where $g$ is a function that combines several factors related to the vulnerability's exploitability and impact. In our synthetic data, 'Known_Vulnerabilities_Score' directly serves as this simplified score. A higher score indicates a more severe vulnerability.
    """)
    if not st.session_state.ai_bom_df.empty:
        max_vuln_score = st.session_state.ai_bom_df['Known_Vulnerabilities_Score'].max()
        min_vuln_score = st.session_state.ai_bom_df['Known_Vulnerabilities_Score'].min()
        avg_vuln_score = st.session_state.ai_bom_df['Known_Vulnerabilities_Score'].mean()

        st.metric(label="Maximum Known Vulnerabilities Score", value=f"{max_vuln_score:.2f}")
        st.metric(label="Minimum Known Vulnerabilities Score", value=f"{min_vuln_score:.2f}")
        st.metric(label="Average Known Vulnerabilities Score", value=f"{avg_vuln_score:.2f}")
```

In our synthetic AI-BOM, the `Known_Vulnerabilities_Score` column acts as this simplified CVSS-like score. A higher value indicates a more severe vulnerability. This score is a direct input for calculating the individual component risk.

## 7. Calculating Individual Component Risk
Duration: 0:06

Each component has an inherent risk. The `calculate_component_risk` function in `app.py` computes a `Component_Risk_Score` by combining the `Known_Vulnerabilities_Score` with penalties based on the `Origin` (e.g., third-party components might carry higher risk).

### `calculate_component_risk` Function

```python
# app.py excerpt
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
```

This function is applied to each row of the AI-BOM DataFrame immediately after generation:

```python
# app.py excerpt (within the 'Generate AI-BOM' button logic)
        st.session_state.ai_bom_df['Component_Risk_Score'] = st.session_state.ai_bom_df.apply(calculate_component_risk, axis=1)
```

The "Component Risk Profile" page then displays these calculated scores.

```python
# application_pages/page_4_component_risk_profile.py excerpt
    st.header("6. Calculating Individual Component Risk")
    st.markdown("""
    Each component in the AI system carries its own set of risks, influenced by attributes like its type, origin, and known vulnerabilities...
    """)
    st.markdown("First 5 components with their vulnerability and calculated risk scores:")
    st.dataframe(st.session_state.ai_bom_df[['Component_ID', 'Component_Name', 'Known_Vulnerabilities_Score', 'Component_Risk_Score']].head())
```

<aside class="positive">
The formula for `calculate_component_risk` is a simplified example. In real-world scenarios, risk models can be far more complex, incorporating factors like criticality, attack vector, exploitability, impact (confidentiality, integrity, availability), and organizational context.
</aside>

## 8. Displaying Component Risk Profiles
Duration: 0:04

To make the risk information easily digestible for risk managers, the application generates a human-readable textual risk profile for selected components.

### `get_component_risk_profile` Function

This function, defined in `app.py` (and duplicated for convenience in `application_pages/page_4_component_risk_profile.py`), constructs a detailed summary for a given component.

```python
# app.py excerpt
def get_component_risk_profile(components_df: pd.DataFrame, component_id: str) -> str:
    """Generates a human-readable textual risk profile for a specified component."""
    component_data = components_df[components_df['Component_ID'] == component_id].iloc[0]

    profile = f"""
Component Risk Profile for: {component_data['Component_Name']} ({component_data['Component_ID']})
--
Component Type: {component_data['Component_Type']}
Origin: {component_data['Origin']}
Version: {component_data['Version']}
Known Vulnerabilities Score: {component_data['Known_Vulnerabilities_Score']:.2f}
Licensing Info: {component_data['Licensing_Info']}
Calculated Component Risk Score: {component_data['Component_Risk_Score']:.2f}
--
"""
    return profile
```

### Interactive Profile Display
On the "Component Risk Profile" page, a `selectbox` allows users to pick any component and view its detailed risk profile.

```python
# application_pages/page_4_component_risk_profile.py excerpt
    st.header("7. Displaying Component Risk Profiles")
    st.markdown("""
    A textual 'Risk Profile' provides a concise summary of a component's key risk-relevant information...
    """)
    
    selected_component_id_profile = st.selectbox(
        "Select a component to view its Risk Profile:",
        options=st.session_state.ai_bom_df['Component_ID'].tolist(),
        key='component_profile_selector'
    )
    if selected_component_id_profile:
        risk_profile = get_component_risk_profile(st.session_state.ai_bom_df, selected_component_id_profile)
        st.text(risk_profile)
```

## 9. Aggregating Overall AI System Risk
Duration: 0:07

The overall AI system risk is not merely a sum of individual component risks. It must also consider how vulnerabilities can propagate through the system's dependencies. A high-risk component with many downstream dependencies can have a far greater impact than a high-risk component that is isolated.

### `aggregate_overall_ai_system_risk` Function

This function in `app.py` calculates a weighted sum, where components with a higher 'out-degree' (more downstream dependencies) contribute more significantly to the overall risk if their individual risk is high.

```python
# app.py excerpt
def aggregate_overall_ai_system_risk(graph: nx.Graph) -> float:
    """Calculates the overall risk for the entire AI system based on individual component risks and interdependencies."""
    total_weighted_risk = 0.0
    max_possible_weighted_risk = 0.0

    for node_id in graph.nodes():
        component_risk_score = graph.nodes[node_id].get('Component_Risk_Score', 0.0)
        out_degree = graph.out_degree(node_id)

        # Weighted risk increases with component risk and its number of downstream dependencies
        weighted_risk = component_risk_score * (1 + out_degree)
        total_weighted_risk += weighted_risk

        # Calculate max possible weighted risk for normalization
        max_component_risk_possible = 15.0 # Defined max from calculate_component_risk
        max_out_degree_possible = len(graph.nodes()) - 1 # Max possible out-degree
        max_possible_weighted_risk += max_component_risk_possible * (1 + max_out_degree_possible)

    if max_possible_weighted_risk == 0:
        return 0.0

    normalized_overall_risk = (total_weighted_risk / max_possible_weighted_risk) * 100
    return round(normalized_overall_risk, 2)
```

The formula for the weighted risk contribution of a single component can be expressed as:
$$ \text{Weighted Risk}_i = \text{Component Risk Score}_i \times (1 + \text{Out-degree}_i) $$
The total weighted risk is the sum of these for all components. This sum is then normalized against the maximum possible weighted risk to produce a score between 0 and 100.

```python
# application_pages/page_4_component_risk_profile.py excerpt
    st.header("8. Aggregating Overall AI System Risk")
    st.markdown("""
    The overall AI System Risk is not just the sum of individual component risks. It must also account for how vulnerabilities can propagate through dependencies...
    $$ \text{Overall AI System Risk} = f(\text{ComponentRisk}_1, \dots, \text{ComponentRisk}_N, \text{Interdependencies}) $$
    """)
    overall_risk_score = aggregate_overall_ai_system_risk(st.session_state.ai_system_graph)
    st.metric(label="Overall Calculated AI System Risk Score (0-100)", value=f"{overall_risk_score:.2f}")
```

### Flowchart for Risk Calculation and Aggregation

```mermaid
graph TD
    A[Start: AI-BOM DataFrame] --> B{Calculate Component Risk Score};
    B -- For each Component --> C[Component Risk Score (CRS)];
    C --> D[Create Directed Graph (Nodes: Components + CRS, Edges: Dependencies)];
    D -- For each Node --> E{Get Out-degree};
    E --> F[Calculate Weighted Risk = CRS * (1 + Out-degree)];
    F --> G[Sum Weighted Risks for All Components];
    G --> H[Calculate Max Possible Weighted Risk];
    H --> I[Normalize Overall Risk = (Sum Weighted Risks / Max Possible Weighted Risk) * 100];
    I --> J[End: Overall AI System Risk Score (0-100)];
```

## 10. Simulating Vulnerability Propagation
Duration: 0:10

One of the most powerful features of the AI-BOM Risk Navigator is its ability to simulate the impact of a newly discovered vulnerability. This allows risk managers to understand potential cascading effects and identify critical propagation paths.

### `simulate_vulnerability_propagation` Function

This function, defined in `app.py`, takes the current graph, a `vulnerable_component_id`, and a `base_impact_score`. It then updates the risk scores of the vulnerable component and its direct and indirect downstream dependencies.

```python
# app.py excerpt
def simulate_vulnerability_propagation(graph: nx.Graph, vulnerable_component_id: str, base_impact_score: float) -> nx.Graph:
    """Simulates a vulnerability in a specified component, propagates its impact, and updates risk scores."""
    simulated_graph = graph.copy() # Create a deep copy to avoid modifying the original graph

    # 1. Directly update the vulnerable component
    if vulnerable_component_id in simulated_graph.nodes:
        component_data_series = pd.Series(simulated_graph.nodes[vulnerable_component_id])
        component_data_series['Known_Vulnerabilities_Score'] = min(component_data_series['Known_Vulnerabilities_Score'] + base_impact_score, 15.0)
        updated_risk_score = calculate_component_risk(component_data_series)
        simulated_graph.nodes[vulnerable_component_id]['Component_Risk_Score'] = updated_risk_score
        simulated_graph.nodes[vulnerable_component_id]['Known_Vulnerabilities_Score'] = component_data_series['Known_Vulnerabilities_Score']
    else:
        st.warning(f"Vulnerable component '{vulnerable_component_id}' not found in the graph.")
        return simulated_graph

    # 2. Propagate attenuated impact to direct downstream dependencies
    for successor in simulated_graph.successors(vulnerable_component_id):
        if successor in simulated_graph.nodes:
            current_risk = simulated_graph.nodes[successor].get('Component_Risk_Score', 0.0)
            propagated_impact = base_impact_score * 0.5 # Direct impact is 50% of base
            simulated_graph.nodes[successor]['Component_Risk_Score'] = min(current_risk + propagated_impact, 15.0)

            current_vuln_score = simulated_graph.nodes[successor].get('Known_Vulnerabilities_Score', 0.0)
            simulated_graph.nodes[successor]['Known_Vulnerabilities_Score'] = min(current_vuln_score + (base_impact_score * 0.2), 15.0)

            # 3. Propagate further attenuated impact to indirect downstream dependencies
            for indirect_successor in simulated_graph.successors(successor):
                if indirect_successor in simulated_graph.nodes and indirect_successor != vulnerable_component_id:
                    current_risk_indirect = simulated_graph.nodes[indirect_successor].get('Component_Risk_Score', 0.0)
                    propagated_impact_indirect = base_impact_score * 0.2 # Indirect impact is 20% of base
                    simulated_graph.nodes[indirect_successor]['Component_Risk_Score'] = min(current_risk_indirect + propagated_impact_indirect, 15.0)

                    current_vuln_score_indirect = simulated_graph.nodes[indirect_successor].get('Known_Vulnerabilities_Score', 0.0)
                    simulated_graph.nodes[indirect_successor]['Known_Vulnerabilities_Score'] = min(current_vuln_score_indirect + (base_impact_score * 0.1), 15.0)

    return simulated_graph
```

### Interacting with Simulation Controls
In the sidebar, after generating the AI-BOM, you can select a component to make vulnerable and specify a `Base Impact Score`.

```python
# app.py excerpt (sidebar section)
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
                current_graph_for_simulation = create_ai_system_graph(st.session_state.ai_bom_df, st.session_state.ai_bom_dependencies)
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
```

## 11. Visualizing Vulnerability Impact
Duration: 0:08

After running a simulation, the "Vulnerability Impact" page (`application_pages/page_5_vulnerability_impact.py`) visualizes the resulting graph. Nodes are now color-coded and sized based on their updated `Component_Risk_Score`, immediately highlighting the most affected areas and the paths of risk propagation.

```python
# application_pages/page_5_vulnerability_impact.py excerpt
def main():
    st.header("9. Simulating a Vulnerability")
    if not st.session_state.simulated_risk_graph.empty():
        st.markdown(f"Simulation performed: Component **'{st.session_state.vulnerable_component_id_input}'** was affected with a base impact score of **{st.session_state.base_impact_score_input}**. Risks have been propagated.")

        st.header("10. Visualizing Vulnerability Propagation")
        st.markdown("""
        Visualizing the impact of a simulated vulnerability clearly demonstrates cascading effects...
        """)
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(st.session_state.simulated_risk_graph, seed=42)

        risk_scores = [st.session_state.simulated_risk_graph.nodes[node].get('Component_Risk_Score', 0.0) for node in st.session_state.simulated_risk_graph.nodes()]
        
        cmap = plt.cm.get_cmap('RdYlGn_r') # Red-Yellow-Green reversed colormap
        norm = plt.Normalize(vmin=0, vmax=15)
        node_colors = [cmap(norm(score)) for score in risk_scores]

        node_sizes = [score * 200 + 1000 for score in risk_scores] # Larger nodes for higher risk

        nx.draw_networkx_nodes(st.session_state.simulated_risk_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(st.session_state.simulated_risk_graph, pos, edgelist=st.session_state.simulated_risk_graph.edges(), edge_color='gray', arrowsize=20, ax=ax)
        nx.draw_networkx_labels(st.session_state.simulated_risk_graph, pos, font_size=8, font_weight='bold', ax=ax)

        ax.set_title("AI System Graph with Simulated Vulnerability Impact")
        ax.axis('off')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(risk_scores)
        cbar = plt.colorbar(sm, orientation='vertical', pad=0.02, ax=ax)
        cbar.set_label('Component Risk Score (0-15)')
        st.pyplot(fig)
    else:
        st.info("Run a vulnerability simulation using the sidebar controls to visualize the impact.")
```

This visualization is a critical tool for incident response planning, allowing risk managers to identify which components are most exposed and prioritize mitigation strategies effectively.

## 12. Targeted Risk Analysis - Data Provenance and Integrity Risks
Duration: 0:06

The "Targeted Risk Analysis" page (`application_pages/page_6_targeted_risk_analysis.py`) provides specialized insights. One key area is assessing risks related to data components. Data provenance (origin) and integrity are fundamental to trustworthy AI.

```python
# application_pages/page_6_targeted_risk_analysis.py excerpt
def main():
    st.header("11. Assessing Data Provenance and Integrity Risks")
    st.markdown("""
    Data provenance and integrity are foundational to trustworthy AI. Compromised or poorly sourced data can lead to biased models, security vulnerabilities, and unreliable predictions.
    Risks associated with data include:
    *   **Data Poisoning**: Malicious data introduced into training sets.
    *   **Data Drift**: Changes in input data distribution over time.
    *   **Bias**: Unfair representation leading to discriminatory outcomes.
    *   **Privacy Violations**: Sensitive information leakage.
    """)
    if not st.session_state.ai_bom_df.empty:
        data_components_df = st.session_state.ai_bom_df[st.session_state.ai_bom_df['Component_Type'] == 'Data'].copy()
        st.subheader("Data Components and their Risk-Relevant Attributes:")
        st.dataframe(data_components_df[['Component_ID', 'Origin', 'Licensing_Info', 'Component_Risk_Score']])
        if not data_components_df.empty:
            mean_data_risk = data_components_df['Component_Risk_Score'].mean()
            st.write(f"Average Component Risk Score for 'Data' components: **{mean_data_risk:.2f}**")
        else:
            st.info("No 'Data' components found in the current AI-BOM.")
```

This section filters the AI-BOM to show only components of `Component_Type == 'Data'` and displays their relevant attributes, allowing for a focused analysis of data-related risks. The `Origin` and `Licensing_Info` attributes are particularly important here.

## 13. Targeted Risk Analysis - Third-Party & Critical Components
Duration: 0:06

Another critical aspect of AI supply chain risk management is evaluating third-party components. These introduce external dependencies and can carry inherent risks. The "Targeted Risk Analysis" page also focuses on these.

```python
# application_pages/page_6_targeted_risk_analysis.py excerpt
    st.header("12. Evaluating Third-Party Model and Component Risks")
    st.markdown("""
    Third-party models, libraries, and hardware introduce external dependencies into the AI system supply chain.
    Risks include:
    *   **Vulnerabilities in third-party code**: Unpatched CVEs in libraries.
    *   **Supply chain attacks**: Malicious code injected into open-source components.
    *   **Model opacity**: Difficulty in auditing pre-trained models.
    *   **Licensing compliance**: Legal risks from non-compliant usage.
    """)
    third_party_origins = ['Third-Party Vendor A', 'Third-Party Vendor B', 'Open Source Community']
    third_party_components_df = st.session_state.ai_bom_df[st.session_state.ai_bom_df['Origin'].isin(third_party_origins)].copy()
    st.subheader("Third-Party Components and their Risk-Relevant Attributes:")
    st.dataframe(third_party_components_df[['Component_ID', 'Component_Type', 'Origin', 'Known_Vulnerabilities_Score', 'Component_Risk_Score']])
    if not third_party_components_df.empty:
        mean_third_party_risk = third_party_components_df['Component_Risk_Score'].mean()
        st.write(f"Average Component Risk Score for third-party components: **{mean_third_party_risk:.2f}**")
    else:
        st.info("No 'Third-Party' components found in the current AI-BOM.")
```

This part filters for components with `Origin` categorized as third-party, providing a focused view on these external dependencies.

Additionally, the page allows you to identify the top and bottom N components by risk score, enabling prioritization of mitigation efforts.

```python
# application_pages/page_6_targeted_risk_analysis.py excerpt
    st.subheader("Top and Bottom N Components by Risk Score")
    st.session_state.num_top_bottom_n_input = st.number_input(
        "Enter N for Top/Bottom Components:", 
        min_value=1, max_value=len(st.session_state.ai_bom_df), value=st.session_state.num_top_bottom_n_input, key='top_bottom_n_input'
    )

    if not st.session_state.ai_bom_df.empty:
        st.markdown(f"Top {st.session_state.num_top_bottom_n_input} components with the highest risk scores:")
        st.dataframe(st.session_state.ai_bom_df.sort_values(by='Component_Risk_Score', ascending=False).head(st.session_state.num_top_bottom_n_input)[['Component_ID', 'Component_Name', 'Component_Type', 'Origin', 'Component_Risk_Score']])

        st.markdown(f"\nBottom {st.session_state.num_top_bottom_n_input} components with the lowest risk scores:")
        st.dataframe(st.session_state.ai_bom_df.sort_values(by='Component_Risk_Score', ascending=True).head(st.session_state.num_top_bottom_n_input)[['Component_ID', 'Component_Name', 'Component_Type', 'Origin', 'Component_Risk_Score']])
```

This feature allows for quick identification of the most critical components requiring immediate attention and those with low risk, aiding in resource allocation.

## 14. Conclusion
Duration: 0:02
This codelab has provided a comprehensive walkthrough of the AI-BOM Risk Navigator, demonstrating how an AI Bill of Materials can be used to understand, visualize, and manage risks within complex AI systems.

Key takeaways include:
*   The fundamental importance of AI-BOM for transparency and risk assessment in AI supply chains.
*   Methods for generating synthetic AI-BOM data and understanding its attributes.
*   Techniques for visualizing AI system dependencies to identify structural risks.
*   Practical approaches to calculating individual component risks and aggregating overall system risk.
*   The power of vulnerability simulation to predict cascading impacts and inform incident response.
*   Targeted analyses for specific risk categories like data provenance and third-party components.

By applying these concepts, developers and risk managers can gain invaluable insights, enabling them to build, deploy, and maintain AI systems that are not only performant but also secure, resilient, and trustworthy. The proactive identification and management of risks, facilitated by tools like the AI-BOM Risk Navigator, are essential for navigating the evolving landscape of AI ethics and security.
