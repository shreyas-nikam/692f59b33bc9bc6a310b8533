
# Streamlit Application Specification: AI-BOM Risk Navigator

## 1. Application Overview

The AI-BOM Risk Navigator is an interactive Streamlit application designed for Risk Managers to explore and assess risks within AI system supply chains. It provides a visual and data-driven approach to understanding component dependencies, identifying vulnerabilities, and simulating the cascading impact of compromised elements.

**Learning Goals:**
- Understand the concept and importance of an AI Bill of Materials (AI-BOM) for AI risk management.
- Learn how to generate and interpret a synthetic AI-BOM dataset.
- Visualize AI system dependencies and their structural integrity.
- Calculate and interpret individual component risk scores based on attributes like vulnerabilities and origin.
- Aggregate overall AI System Risk, accounting for component interdependencies.
- Simulate the propagation of a vulnerability throughout the AI system and visualize its impact.
- Assess specific risks related to data provenance and third-party components.
- Identify top and bottom risk components to prioritize mitigation efforts.

## 2. User Interface Requirements

The application will feature a logical flow, with interactive controls primarily located in a sidebar and analytical outputs displayed in the main content area, possibly organized into tabs or expandable sections.

### Layout and Navigation Structure

-   **Sidebar (`st.sidebar`):** Will host global configurations and inputs for generating the AI-BOM and initiating simulations.
    -   Controls for AI-BOM generation (`num_components`, `num_dependencies`).
    -   Controls for vulnerability simulation (`vulnerable_component_id`, `base_impact_score`).
    -   A "Generate AI-BOM" button and a "Run Simulation" button.
-   **Main Content Area:** Will be divided into logical sections or tabs to guide the user through the AI-BOM analysis.
    -   **Introduction:** Initial welcome and explanation of AI-BOM.
    -   **AI-BOM Details:** Display of the generated AI-BOM DataFrame, its information, and descriptive statistics.
    -   **Initial Dependency Graph:** Visualization of the AI system before any simulation.
    -   **Component Risk Assessment:** Section to display component risk profiles upon selection, vulnerability score statistics, and overall AI system risk.
    -   **Vulnerability Simulation:** Visualization of the graph with propagated risk and associated metrics.
    -   **Targeted Risk Analysis:** Sections for 'Data' and 'Third-Party' components, and top/bottom risk components.

### Input Widgets and Controls

1.  **AI-BOM Generation:**
    -   `st.slider` or `st.number_input` for `Number of Components` (e.g., range 5-50, default 15).
    -   `st.slider` or `st.number_input` for `Number of Dependencies` (e.g., range 5-100, default 20).
    -   `st.button` labeled "Generate AI-BOM" to trigger dataset creation and initial graph construction.
2.  **Component Risk Profile Selection:**
    -   `st.selectbox` labeled "Select Component ID" populated with `Component_ID`s from the `ai_bom_df`. Selecting a component will display its detailed risk profile.
3.  **Vulnerability Simulation:**
    -   `st.selectbox` labeled "Vulnerable Component ID" populated with `Component_ID`s from the current AI-BOM, allowing users to choose the component to compromise.
    -   `st.slider` or `st.number_input` for `Base Impact Score` (e.g., range 0-10, default 7.0), representing the severity of the simulated vulnerability.
    -   `st.button` labeled "Run Vulnerability Simulation" to initiate the propagation.
4.  **Top/Bottom Components Analysis:**
    -   `st.number_input` labeled "Number of Components (N)" for displaying top/bottom N components (e.g., default 3, min 1, max 10).

### Visualization Components

1.  **AI-BOM DataFrame:**
    -   `st.dataframe` to display the `ai_bom_df`.
    -   `st.text` or `st.code` to display `ai_bom_df.info()` and `ai_bom_df.describe()`.
2.  **Vulnerability Statistics:**
    -   `st.metric` or `st.write` to display `Maximum Known Vulnerabilities Score`, `Minimum Known Vulnerabilities Score`, and `Average Known Vulnerabilities Score`.
3.  **Initial AI System Dependency Graph:**
    -   `st.pyplot` to render the NetworkX graph.
    -   Nodes color-coded by `Component_Type` with a clear legend.
    -   Node labels showing `Component_ID` or `Component_Name`.
    -   Fixed `nx.spring_layout` seed for reproducible layout.
4.  **Component Risk Profile:**
    -   `st.text_area` or `st.markdown` to display the human-readable risk profile of the selected component.
5.  **Overall AI System Risk Score:**
    -   `st.metric` or `st.markdown` for prominent display of the `Overall AI System Risk Score` (normalized 0-100).
6.  **Simulated Vulnerability Propagation Graph:**
    -   `st.pyplot` to render the NetworkX graph after simulation.
    -   Nodes color-coded (e.g., green-to-red gradient) based on `Component_Risk_Score`.
    -   Node sizes proportional to `Component_Risk_Score`.
    -   A vertical color bar indicating the `Component_Risk_Score` range.
7.  **Targeted Component Analysis Tables:**
    -   `st.dataframe` for 'Data' components, showing relevant attributes and average risk score.
    -   `st.dataframe` for 'Third-Party' components, showing relevant attributes and average risk score.
    -   `st.dataframe` for top N highest risk components.
    -   `st.dataframe` for bottom N lowest risk components.

### Interactive Elements and Feedback Mechanisms

-   **Button-triggered actions:** "Generate AI-BOM" and "Run Vulnerability Simulation" buttons will re-run respective sections of the application, updating displays.
-   **Dropdown-triggered updates:** Selecting a `Component_ID` from the dropdown will instantly display its risk profile.
-   **Real-time score updates:** The `Overall AI System Risk Score` will update immediately after a simulation.
-   **Progress indicators:** `st.spinner` could be used for longer operations like graph generation or simulation.

## 3. Additional Requirements

### Annotation and Tooltip Specifications

-   **Initial Graph:** A legend for component types will be provided. Node labels will display `Component_ID` or `Component_Name`.
-   **Simulated Graph:** A color bar will clearly indicate the mapping of node colors to `Component_Risk_Score` values. Node labels will display `Component_ID` or `Component_Name`.
-   **Component Risk Profile:** Upon selection of a component ID, a detailed textual summary will be displayed, outlining all relevant attributes and calculated risks.

### Save the States of the Fields Properly

-   The application will leverage `st.session_state` to persist crucial data and user inputs across reruns, ensuring a seamless interactive experience. This includes:
    -   `ai_bom_df` (the generated component DataFrame).
    -   `ai_bom_dependencies` (the list of dependencies).
    -   `ai_system_graph` (the initial NetworkX graph).
    -   `simulated_risk_graph` (the NetworkX graph after simulation).
    -   User-defined parameters like `num_components`, `num_dependencies`, `vulnerable_component_id`, `base_impact_score`, and `num_top_bottom_n`.
-   This approach prevents loss of data when interacting with different widgets or sections of the application.

## 4. Notebook Content and Code Requirements

This section outlines how the content and code from the Jupyter notebook will be integrated into the Streamlit application, preserving all markdown and utilizing the provided Python functions.

```python
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import streamlit as st

# Configure plot styles for better readability (will be applied once at app start)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100
```

### Application Structure Overview

The Streamlit app will be structured as follows:

```python
st.set_page_config(layout="wide", page_title="AI-BOM Risk Navigator")
st.title("AI-BOM Risk Navigator: Navigating Risks in AI Systems")

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

# Main content area tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Introduction", "AI-BOM Details", "Initial Dependencies", 
    "Component Risk Profile", "Vulnerability Impact", "Targeted Risk Analysis"
])

with tab1:
    st.header("1. Introduction to AI Bill of Materials (AI-BOM) for Risk Management")
    st.markdown("""
    An AI Bill of Materials (AI-BOM) provides a structured inventory of all components (datasets, models, libraries, hardware) that comprise an AI system, along with their dependencies. This transparency is crucial for managing risks across the AI supply chain.

    For a complex AI system, understanding dependencies is paramount. Just as a software Bill of Materials (SBOM) tracks software components, an AI-BOM extends this concept to cover AI-specific elements like training data and model architectures. This helps in identifying potential vulnerabilities, assessing data provenance, and understanding the cascading impact of a compromised element.
    """)
    st.markdown("---")
    st.markdown("""
    This section lays the foundation for understanding what an AI-BOM is and why it's essential for AI risk management. Subsequent sections will build upon this by generating an AI-BOM, visualizing it, and analyzing risks.
    """)

with tab2:
    st.header("2. Generating a Synthetic AI-BOM Dataset")
    st.markdown("""
    To demonstrate the utility of an AI-BOM, we will generate a synthetic dataset. This dataset will represent various components of an AI system and their interdependencies. Each component will have attributes crucial for risk assessment, such as type, origin, version, known vulnerabilities, and licensing information. The dependencies will define the flow and connections between these components.
    """)
    if not st.session_state.ai_bom_df.empty:
        st.subheader("AI-BOM DataFrame")
        st.dataframe(st.session_state.ai_bom_df)
        st.markdown("""
        The generated synthetic AI-BOM data provides a realistic foundation for our analysis. We can now see the various components that constitute our hypothetical AI system and their initial risk-relevant attributes. This table is the raw input for building our dependency graph.
        """)

        st.subheader("3. Understanding Component Attributes")
        st.markdown("""
        Each entry in the AI-BOM describes a component with several key attributes. These attributes are critical for assessing individual component risks and their potential impact on the overall AI system.
        *   **Component Name/ID**: A unique identifier for the component.
        *   **Component Type**: Categorizes the component (e.g., 'Data', 'Model', 'Library', 'Hardware'). Different types may have different risk profiles.
        *   **Origin/Provenance**: Where the component came from (e.g., 'Internal', 'Third-Party Vendor A', 'Open Source Community'). Crucial for assessing supply chain risks.
        *   **Version**: Specific version details, important for tracking known vulnerabilities.
        *   **Known Vulnerabilities (Score)**: A numerical representation of identified security weaknesses. We use a simplified score, conceptually related to CVSS.
        *   **Licensing Information**: Details about the license, which can imply legal or security risks.

        These attributes, especially 'Known_Vulnerabilities_Score' and 'Origin', will directly influence a component's risk profile.
        """)
        st.markdown("### AI-BOM DataFrame Information:")
        info_buffer = io.StringIO()
        st.session_state.ai_bom_df.info(buf=info_buffer)
        st.text(info_buffer.getvalue())

        st.markdown("### AI-BOM DataFrame Descriptive Statistics:")
        st.dataframe(st.session_state.ai_bom_df.describe())
        st.markdown("""
        By examining the `ai_bom_df`'s attributes, we get a clearer picture of the data we're working with. Understanding these attributes is the first step in identifying potential risk factors associated with each component in our AI system.
        """)
    else:
        st.info("Please generate an AI-BOM using the sidebar controls to see the dataset and its attributes.")

with tab3:
    st.header("4. Visualizing AI System Dependencies (Initial Graph)")
    st.markdown("""
    A graphical representation of the AI system, where components are nodes and dependencies are edges, significantly enhances our understanding of its structure. This visualization allows Risk Managers to quickly grasp the complexity and interconnections within the AI supply chain.
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
        st.markdown("""
        The visual graph provides an immediate understanding of the AI system's architecture. We can see how different components are interconnected, which is fundamental for identifying critical paths and potential single points of failure. The color-coding by component type gives a quick overview of the system's composition.
        """)
    else:
        st.info("Generate an AI-BOM to visualize the initial dependency graph.")

with tab4:
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
        st.markdown("""
        Understanding the range and distribution of vulnerability scores in our AI-BOM gives us a baseline for assessing the security posture of individual components. This numerical value is a direct input for our component risk calculations.
        """)

        st.header("6. Calculating Individual Component Risk")
        st.markdown("""
        Each component in the AI system carries its own set of risks, influenced by attributes like its type, origin, and known vulnerabilities. To assess these individual risks, we combine these attributes into a single 'Component Risk' score. For simplicity, we'll define a risk function that considers the 'Known_Vulnerabilities_Score' and potentially the 'Origin' (e.g., third-party components might inherently carry higher risk).
        """)
        st.markdown("First 5 components with their vulnerability and calculated risk scores:")
        st.dataframe(st.session_state.ai_bom_df[['Component_ID', 'Component_Name', 'Known_Vulnerabilities_Score', 'Component_Risk_Score']].head())
        st.markdown("""
        We have now quantified the individual risk associated with each component. This 'Component_Risk_Score' provides a granular view of where potential problems might lie within the AI system, taking into account both reported vulnerabilities and supply chain factors.
        """)

        st.header("7. Displaying Component Risk Profiles")
        st.markdown("""
        A textual 'Risk Profile' provides a concise summary of a component's key risk-relevant information. This allows Risk Managers to quickly understand the implications of a specific component's presence in the AI system without needing to parse raw data. The profile should highlight origin, vulnerabilities, and the calculated component risk score.
        """)
        
        selected_component_id_profile = st.selectbox(
            "Select a component to view its Risk Profile:",
            options=st.session_state.ai_bom_df['Component_ID'].tolist(),
            key='component_profile_selector'
        )
        if selected_component_id_profile:
            risk_profile = get_component_risk_profile(st.session_state.ai_bom_df, selected_component_id_profile)
            st.text(risk_profile)
        st.markdown("""
        The component risk profile offers a quick and comprehensive summary, allowing Risk Managers to efficiently assess specific components. This is crucial for focused investigations and for understanding the details behind a component's assigned risk score.
        """)

        st.header("8. Aggregating Overall AI System Risk")
        st.markdown("""
        The overall AI System Risk is not just the sum of individual component risks. It must also account for how vulnerabilities can propagate through dependencies. A critical vulnerability in a foundational library, for example, could impact multiple models or data processing steps that depend on it.

        We define the overall risk aggregation function as:
        $$ \text{Overall AI System Risk} = f(\text{ComponentRisk}_1, \dots, \text{ComponentRisk}_N, \text{Interdependencies}) $$
        where $f$ is an aggregation function. A simple aggregation might be the maximum risk score found in any component, or a weighted average that gives more weight to components with many downstream dependencies. For this lab, we will use a weighted sum, where components with higher 'out-degrees' (more downstream dependencies) contribute more significantly to the overall risk if their individual risk is high.
        """)
        overall_risk_score = aggregate_overall_ai_system_risk(st.session_state.ai_system_graph)
        st.metric(label="Overall Calculated AI System Risk Score (0-100)", value=f"{overall_risk_score:.2f}")
        st.markdown("""
        By aggregating the individual component risks and considering their interdependencies, we arrive at a more holistic view of the AI system's security posture. This overall score helps in strategic decision-making regarding risk tolerance and resource allocation for mitigation efforts.
        """)
    else:
        st.info("Generate an AI-BOM to see vulnerability scoring, component risks, and overall system risk.")

with tab5:
    st.header("9. Simulating a Vulnerability")
    st.markdown("""
    Simulating a vulnerability is a powerful way to understand potential impacts and identify critical risk propagation paths. We will select a component and artificially increase its 'Known_Vulnerabilities_Score' to reflect a newly discovered, severe exploit (e.g., a zero-day exploit). This simulation will then trigger a re-evaluation of risks throughout the system.
    """)
    if not st.session_state.simulated_risk_graph.empty():
        st.markdown(f"Simulation performed: Component **'{st.session_state.vulnerable_component_id_input}'** was affected with a base impact score of **{st.session_state.base_impact_score_input}**. Risks have been propagated.")
        st.markdown("""
        By simulating a critical vulnerability, we've created a scenario to observe how such an event could escalate. This modified graph is now ready to show us the ripple effect across the entire AI system, which is invaluable for pre-emptive planning and incident response.
        """)

        st.header("10. Visualizing Vulnerability Propagation")
        st.markdown("""
        Visualizing the impact of a simulated vulnerability clearly demonstrates cascading effects. By comparing the graph before and after the simulation, we can identify which components are most affected and how risks propagate through the system's dependencies. Nodes will be color-coded based on their updated risk scores to immediately highlight highly impacted areas.
        """)
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(st.session_state.simulated_risk_graph, seed=42)

        risk_scores = [st.session_state.simulated_risk_graph.nodes[node].get('Component_Risk_Score', 0.0) for node in st.session_state.simulated_risk_graph.nodes()]
        
        cmap = plt.cm.get_cmap('RdYlGn_r')
        norm = plt.Normalize(vmin=0, vmax=15)
        node_colors = [cmap(norm(score)) for score in risk_scores]

        node_sizes = [score * 200 + 1000 for score in risk_scores]

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
        st.markdown("""
        This visualization offers a critical insight into the system's resilience and potential vulnerabilities. Risk Managers can clearly see which parts of the AI system are most exposed to a threat originating from a specific component, enabling them to prioritize mitigation strategies for critical paths.
        """)
    else:
        st.info("Run a vulnerability simulation using the sidebar controls to visualize the impact.")

with tab6:
    st.header("11. Assessing Data Provenance and Integrity Risks")
    st.markdown("""
    Data provenance and integrity are foundational to trustworthy AI. Compromised or poorly sourced data can lead to biased models, security vulnerabilities, and unreliable predictions. The AI-BOM helps track the origin and characteristics of data components, allowing for focused risk assessment.
    Risks associated with data include:
    *   **Data Poisoning**: Malicious data introduced into training sets.
    *   **Data Drift**: Changes in input data distribution over time.
    *   **Bias**: Unfair representation leading to discriminatory outcomes.
    *   **Privacy Violations**: Sensitive information leakage.

    For data components, the 'Origin' and 'Licensing_Info' attributes in our AI-BOM are particularly relevant for assessing these risks.
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
        st.markdown("""
        By isolating and examining data components, we can specifically identify and address risks related to data quality, origin, and integrity. This focused analysis supports the development of strategies for robust data governance and provenance tracking.
        """)

        st.header("12. Evaluating Third-Party Model and Component Risks")
        st.markdown("""
        Third-party models, libraries, and hardware introduce external dependencies into the AI system supply chain. These components can carry inherent risks due to lack of transparency, unknown biases, or unpatched vulnerabilities. Rigorous due diligence and continuous monitoring are essential.

        Risks include:
        *   **Vulnerabilities in third-party code**: Unpatched CVEs in libraries.
        *   **Supply chain attacks**: Malicious code injected into open-source components.
        *   **Model opacity**: Difficulty in auditing pre-trained models.
        *   **Licensing compliance**: Legal risks from non-compliant usage.

        The 'Origin' and 'Known_Vulnerabilities_Score' attributes are particularly important here.
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
        st.markdown("""
        This targeted analysis of third-party components highlights their contribution to the overall risk profile. Understanding these external risks is vital for implementing robust vetting processes, contractual agreements, and continuous monitoring for components originating outside the organization's direct control.
        """)

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
        st.markdown("""
        By identifying the highest and lowest risk components, we can understand where to focus our vulnerability management efforts. High-risk components, especially those from third-parties or with many dependencies, warrant immediate attention for mitigation, patching, or replacement, thereby improving the overall transparency and security of the AI supply chain.
        """)
    else:
        st.info("Generate an AI-BOM to perform data provenance, third-party, and top/bottom component risk analysis.")

st.markdown("---")
st.markdown("""
The insights gained from an AI-BOM are indispensable for maintaining the security, integrity, and trustworthiness of AI deployments in an increasingly complex threat landscape. Proactive risk identification and management are key to building resilient AI systems.
""")
```

### Extracted Code Stubs and Usage

#### Dependencies & Setup

```python
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random
import streamlit as st
import io # Required for ai_bom_df.info() output redirection

# Initial configuration (to be run once at the top of the Streamlit app)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100
```

#### `generate_ai_bom_dataset` Function

```python
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
**Usage in Streamlit:** This function will be called when the "Generate AI-BOM" button in the sidebar is pressed. The `num_components` and `num_dependencies` will be taken from `st.session_state` values set by slider widgets. The returned `components_df` and `dependencies` will be stored in `st.session_state.ai_bom_df` and `st.session_state.ai_bom_dependencies` respectively.

#### `create_ai_system_graph` Function

```python
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
**Usage in Streamlit:** This function will be called immediately after `generate_ai_bom_dataset` is run (upon "Generate AI-BOM" button click) to create the initial graph. It takes `st.session_state.ai_bom_df` and `st.session_state.ai_bom_dependencies` as inputs. The resulting graph will be stored in `st.session_state.ai_system_graph`. It will also be called within the simulation logic to create a fresh graph for simulation.

#### `calculate_component_risk` Function

```python
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
**Usage in Streamlit:** This function will be applied to each row of the `ai_bom_df` to create the 'Component_Risk_Score' column after the AI-BOM is generated. It will also be used internally by `simulate_vulnerability_propagation` to recalculate risk for affected components.

#### `get_component_risk_profile` Function

```python
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
```
**Usage in Streamlit:** This function will be called when a user selects a `Component_ID` from the `st.selectbox` in the "Component Risk Profile" tab. The returned string will be displayed using `st.text`.

#### `aggregate_overall_ai_system_risk` Function

```python
def aggregate_overall_ai_system_risk(graph: nx.Graph) -> float:
    """Calculates the overall risk for the entire AI system based on individual component risks and interdependencies."""
    total_weighted_risk = 0.0
    max_possible_weighted_risk = 0.0

    # Ensure the graph nodes have 'Component_Risk_Score'
    # This loop is already done during initial graph creation and simulation, but as a safeguard:
    # for node_id in graph.nodes():
    #     if 'Component_Risk_Score' not in graph.nodes[node_id]:
    #         # If missing, calculate it (e.g., from original ai_bom_df or default to 0)
    #         # For this spec, we assume it's always present on nodes.
    #         pass 

    for node_id in graph.nodes():
        component_risk_score = graph.nodes[node_id].get('Component_Risk_Score', 0.0)
        out_degree = graph.out_degree(node_id)

        # Weight by out-degree: components influencing more others contribute more
        weighted_risk = component_risk_score * (1 + out_degree)
        total_weighted_risk += weighted_risk

        # Calculate max possible weighted risk for normalization
        max_component_risk_possible = 15.0 # Defined max from calculate_component_risk
        max_out_degree_possible = len(graph.nodes()) - 1
        max_possible_weighted_risk += max_component_risk_possible * (1 + max_out_degree_possible)

    if max_possible_weighted_risk == 0:
        return 0.0

    # Normalize the total weighted risk to a scale of 0-100 for easier interpretation
    normalized_overall_risk = (total_weighted_risk / max_possible_weighted_risk) * 100
    return round(normalized_overall_risk, 2)
```
**Usage in Streamlit:** This function will be called in the "Component Risk Profile" tab, taking `st.session_state.ai_system_graph` as input. The returned score will be displayed using `st.metric`. After a simulation, this function will be called with `st.session_state.simulated_risk_graph` to show the updated overall risk.

#### `simulate_vulnerability_propagation` Function

```python
def simulate_vulnerability_propagation(graph: nx.Graph, vulnerable_component_id: str, base_impact_score: float) -> nx.Graph:
    """Simulates a vulnerability in a specified component, propagates its impact, and updates risk scores."""
    simulated_graph = graph.copy() # Create a deep copy to avoid modifying the original graph

    # 1. Directly update the vulnerable component
    if vulnerable_component_id in simulated_graph.nodes:
        # Get current component data to recalculate risk
        # This assumes the graph nodes contain all necessary attributes from ai_bom_df
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
```
**Usage in Streamlit:** This function will be called when the "Run Vulnerability Simulation" button in the sidebar is pressed. It will take a copy of the current `ai_system_graph` from `st.session_state`, `vulnerable_component_id` from `st.session_state.vulnerable_component_id_input`, and `base_impact_score` from `st.session_state.base_impact_score_input`. The returned `simulated_graph` will be stored in `st.session_state.simulated_risk_graph` for visualization and further analysis.

