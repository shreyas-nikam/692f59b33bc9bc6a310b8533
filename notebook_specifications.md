
# Technical Specification for Jupyter Notebook: AI-BOM Risk Navigator

## 1. Notebook Overview

### Learning Goals
This notebook aims to provide a practical understanding of AI Bill of Materials (AI-BOM) for risk management within AI systems. Upon completion, users will be able to:
1.  Understand the concept and utility of an AI-BOM for risk management.
2.  Identify and track dependencies within complex AI systems.
3.  Assess data provenance and integrity risks.
4.  Evaluate risks associated with third-party models and components.
5.  Develop strategies for supply chain transparency and vulnerability management.

### Target Audience
This notebook is primarily targeted at **Risk Managers**. It will also be valuable for AI Project Leads and Supply Chain Security Analysts seeking to understand and mitigate risks within AI system supply chains.

## 2. Code Requirements

### List of Expected Libraries
The following Python libraries will be used:
*   `pandas` for data manipulation and tabular data representation.
*   `numpy` for numerical operations, especially during synthetic data generation and risk calculations.
*   `networkx` for creating, manipulating, and studying the structure, dynamics, and functions of complex networks (the AI-BOM graph).
*   `matplotlib.pyplot` for generating static, interactive, and animated visualizations in Python.
*   `seaborn` for creating informative and attractive statistical graphics, built on `matplotlib`.

### List of Algorithms or Functions to be Implemented
*   `generate_ai_bom_dataset(num_components: int, num_dependencies: int) -> pd.DataFrame, list[tuple]`: Generates a synthetic AI-BOM dataset (DataFrame) and a list of dependencies (edges).
*   `create_ai_system_graph(components_df: pd.DataFrame, dependencies: list[tuple]) -> nx.Graph`: Constructs a `networkx` graph object representing the AI system from the AI-BOM DataFrame and dependency list.
*   `calculate_component_risk(component_data: pd.Series) -> float`: Computes a risk score for an individual component based on its attributes (e.g., vulnerabilities, origin, type).
*   `get_component_risk_profile(components_df: pd.DataFrame, component_id: str) -> str`: Generates a human-readable textual risk profile for a specified component.
*   `simulate_vulnerability_propagation(graph: nx.Graph, vulnerable_component_id: str, base_impact_score: float) -> nx.Graph`: Simulates a vulnerability in a specified component, propagates its impact through dependencies, and updates risk scores in the graph.
*   `aggregate_overall_ai_system_risk(graph: nx.Graph) -> float`: Calculates the overall risk for the entire AI system based on individual component risks and interdependencies.

### Visualization Requirements
The notebook will generate the following visualizations:
*   **AI-BOM Data Table**: A `pandas` DataFrame displayed as a table, showing the synthetic AI-BOM components and their attributes.
*   **Initial AI System Graph**: A network graph (nodes for components, edges for dependencies) illustrating the complete AI system structure. Nodes will be color-coded by component type (e.g., data, model, library).
*   **Component Dependency Highlight Graph**: A network graph that highlights a selected component and its direct upstream and downstream dependencies.
*   **Component Risk Profile Display**: A formatted markdown output showing detailed risk attributes and a summary for a selected component.
*   **Vulnerability Impact Propagation Graph**: A network graph that visualizes the cascading effects of a simulated vulnerability. Nodes will be color-coded and sized based on their updated risk scores after the simulation.
*   **Overall Risk Score Display**: A clear numerical display of the calculated overall AI system risk.

## 3. Notebook Sections (in detail)

---

### Section 1: Introduction to AI Bill of Materials (AI-BOM) for Risk Management

#### Markdown Cell
An AI Bill of Materials (AI-BOM) provides a structured inventory of all components (datasets, models, libraries, hardware) that comprise an AI system, along with their dependencies. This transparency is crucial for managing risks across the AI supply chain.

For a complex AI system, understanding dependencies is paramount. Just as a software Bill of Materials (SBOM) tracks software components, an AI-BOM extends this concept to cover AI-specific elements like training data and model architectures. This helps in identifying potential vulnerabilities, assessing data provenance, and understanding the cascading impact of a compromised element.

#### Code Cell (Function Implementation Description)
No specific function implementation is required for this introductory section.

#### Code Cell (Function Execution Description)
No specific function execution is required for this introductory section.

#### Markdown Cell
This section lays the foundation for understanding what an AI-BOM is and why it's essential for AI risk management. Subsequent sections will build upon this by generating an AI-BOM, visualizing it, and analyzing risks.

---

### Section 2: Generating a Synthetic AI-BOM Dataset

#### Markdown Cell
To demonstrate the utility of an AI-BOM, we will generate a synthetic dataset. This dataset will represent various components of an AI system and their interdependencies. Each component will have attributes crucial for risk assessment, such as type, origin, version, known vulnerabilities, and licensing information. The dependencies will define the flow and connections between these components.

#### Code Cell (Function Implementation Description)
Implement a Python function `generate_ai_bom_dataset` that creates a synthetic AI-BOM.
*   **Function Name:** `generate_ai_bom_dataset`
*   **Parameters:** `num_components: int`, `num_dependencies: int`
*   **Returns:** `pd.DataFrame` (components_df), `list[tuple]` (dependencies)
*   **Details:**
    *   The `components_df` will have columns: 'Component_ID', 'Component_Name', 'Component_Type' (e.g., 'Data', 'Model', 'Library', 'Hardware'), 'Origin', 'Version', 'Known_Vulnerabilities_Score' (random float between 0-10, representing a simplified CVSS-like score), 'Licensing_Info' (e.g., 'Open Source', 'Proprietary').
    *   `Component_ID` should be unique (e.g., 'Comp_001').
    *   `dependencies` will be a list of tuples `(source_component_id, target_component_id)`. Ensure `num_dependencies` does not exceed the maximum possible edges for the given `num_components` to avoid errors during graph creation.

#### Code Cell (Function Execution Description)
Call `generate_ai_bom_dataset` with `num_components=15` and `num_dependencies=20`.
Store the returned DataFrame as `ai_bom_df` and the list of dependencies as `ai_bom_dependencies`.
Display the first 5 rows of `ai_bom_df` using `ai_bom_df.head()`.

#### Markdown Cell
The generated synthetic AI-BOM data provides a realistic foundation for our analysis. We can now see the various components that constitute our hypothetical AI system and their initial risk-relevant attributes. This table is the raw input for building our dependency graph.

---

### Section 3: Understanding Component Attributes

#### Markdown Cell
Each entry in the AI-BOM describes a component with several key attributes. These attributes are critical for assessing individual component risks and their potential impact on the overall AI system.
*   **Component Name/ID**: A unique identifier for the component.
*   **Component Type**: Categorizes the component (e.g., 'Data', 'Model', 'Library', 'Hardware'). Different types may have different risk profiles.
*   **Origin/Provenance**: Where the component came from (e.g., 'Internal', 'Third-Party Vendor A', 'Open Source Community'). Crucial for assessing supply chain risks.
*   **Version**: Specific version details, important for tracking known vulnerabilities.
*   **Known Vulnerabilities (Score)**: A numerical representation of identified security weaknesses. We use a simplified score, conceptually related to CVSS.
*   **Licensing Information**: Details about the license, which can imply legal or security risks.

These attributes, especially 'Known_Vulnerabilities_Score' and 'Origin', will directly influence a component's risk profile.

#### Code Cell (Function Implementation Description)
No specific function implementation is required for this section, as it's an explanation of the data structure.

#### Code Cell (Function Execution Description)
Display the data types and a summary of the `ai_bom_df` using `ai_bom_df.info()` and `ai_bom_df.describe()`.

#### Markdown Cell
By examining the `ai_bom_df`'s attributes, we get a clearer picture of the data we're working with. Understanding these attributes is the first step in identifying potential risk factors associated with each component in our AI system.

---

### Section 4: Visualizing AI System Dependencies (Initial Graph)

#### Markdown Cell
A graphical representation of the AI system, where components are nodes and dependencies are edges, significantly enhances our understanding of its structure. This visualization allows Risk Managers to quickly grasp the complexity and interconnections within the AI supply chain.

#### Code Cell (Function Implementation Description)
Implement a Python function `create_ai_system_graph` that builds a `networkx` graph.
*   **Function Name:** `create_ai_system_graph`
*   **Parameters:** `components_df: pd.DataFrame`, `dependencies: list[tuple]`
*   **Returns:** `nx.Graph`
*   **Details:**
    *   Create a directed graph (`nx.DiGraph`).
    *   Add nodes using `components_df['Component_ID']`.
    *   For each node, add node attributes from `components_df` columns (e.g., 'Component_Name', 'Component_Type', 'Known_Vulnerabilities_Score').
    *   Add edges from the `dependencies` list.

#### Code Cell (Function Execution Description)
Call `create_ai_system_graph` with `ai_bom_df` and `ai_bom_dependencies`.
Store the returned graph as `ai_system_graph`.
Visualize the graph using `networkx` and `matplotlib.pyplot`.
*   Use `nx.draw_networkx_nodes`, `nx.draw_networkx_edges`, `nx.draw_networkx_labels`.
*   Nodes should be colored based on 'Component_Type'.
*   Include a legend for component types.
*   Set a fixed seed for `nx.spring_layout` for reproducible layouts.

#### Markdown Cell
The visual graph provides an immediate understanding of the AI system's architecture. We can see how different components are interconnected, which is fundamental for identifying critical paths and potential single points of failure. The color-coding by component type gives a quick overview of the system's composition.

---

### Section 5: Introduction to Vulnerability Scoring

#### Markdown Cell
Vulnerability scoring provides a standardized way to quantify the severity of security weaknesses. The Common Vulnerability Scoring System (CVSS) is a widely used open framework for communicating these characteristics and impacts. A simplified version of CVSS helps in prioritizing risks.

The CVSS score is derived from various metrics, broadly categorized into base, temporal, and environmental metrics. For our simplified scenario, we can represent this as:
$$
\text{CVSS Score} = g(\text{AttackVector}, \text{AttackComplexity}, \text{PrivilegesRequired}, \dots)
$$
where $g$ is a function that combines several factors related to the vulnerability's exploitability and impact. In our synthetic data, 'Known_Vulnerabilities_Score' directly serves as this simplified score. A higher score indicates a more severe vulnerability.

#### Code Cell (Function Implementation Description)
No specific function implementation is required for this section, as it's an explanatory concept.

#### Code Cell (Function Execution Description)
Display the maximum, minimum, and average 'Known_Vulnerabilities_Score' from `ai_bom_df`.

#### Markdown Cell
Understanding the range and distribution of vulnerability scores in our AI-BOM gives us a baseline for assessing the security posture of individual components. This numerical value is a direct input for our component risk calculations.

---

### Section 6: Calculating Individual Component Risk

#### Markdown Cell
Each component in the AI system carries its own set of risks, influenced by attributes like its type, origin, and known vulnerabilities. To assess these individual risks, we combine these attributes into a single 'Component Risk' score. For simplicity, we'll define a risk function that considers the 'Known_Vulnerabilities_Score' and potentially the 'Origin' (e.g., third-party components might inherently carry higher risk).

#### Code Cell (Function Implementation Description)
Implement a Python function `calculate_component_risk` that assigns a risk score to each component.
*   **Function Name:** `calculate_component_risk`
*   **Parameters:** `component_data: pd.Series` (a row from `ai_bom_df` for a single component)
*   **Returns:** `float` (the calculated risk score)
*   **Details:**
    *   The risk score will primarily be `component_data['Known_Vulnerabilities_Score']`.
    *   Add a penalty if 'Origin' is 'Third-Party Vendor A' or 'Third-Party Vendor B' (e.g., +2 points to the score).
    *   Ensure the score is between 0 and 10 (or a similar defined range).

#### Code Cell (Function Execution Description)
Apply the `calculate_component_risk` function to each row of `ai_bom_df` to create a new 'Component_Risk_Score' column.
Update the `ai_system_graph` nodes with this new 'Component_Risk_Score' attribute.
Display the 'Component_ID', 'Component_Name', 'Known_Vulnerabilities_Score', and 'Component_Risk_Score' for the first 5 components.

#### Markdown Cell
We have now quantified the individual risk associated with each component. This 'Component_Risk_Score' provides a granular view of where potential problems might lie within the AI system, taking into account both reported vulnerabilities and supply chain factors.

---

### Section 7: Displaying Component Risk Profiles

#### Markdown Cell
A textual 'Risk Profile' provides a concise summary of a component's key risk-relevant information. This allows Risk Managers to quickly understand the implications of a specific component's presence in the AI system without needing to parse raw data. The profile should highlight origin, vulnerabilities, and the calculated component risk score.

#### Code Cell (Function Implementation Description)
Implement a Python function `get_component_risk_profile` to generate a formatted risk profile string.
*   **Function Name:** `get_component_risk_profile`
*   **Parameters:** `components_df: pd.DataFrame`, `component_id: str`
*   **Returns:** `str` (formatted risk profile)
*   **Details:**
    *   Retrieve the component's data from `components_df` using `component_id`.
    *   Format the output to include: Component ID, Name, Type, Origin, Version, Known Vulnerabilities Score, Licensing Info, and the calculated Component Risk Score.

#### Code Cell (Function Execution Description)
Select a specific `component_id` (e.g., 'Comp_005').
Call `get_component_risk_profile` with `ai_bom_df` and the selected `component_id`.
Print the returned risk profile string.

#### Markdown Cell
The component risk profile offers a quick and comprehensive summary, allowing Risk Managers to efficiently assess specific components. This is crucial for focused investigations and for understanding the details behind a component's assigned risk score.

---

### Section 8: Aggregating Overall AI System Risk

#### Markdown Cell
The overall AI System Risk is not just the sum of individual component risks. It must also account for how vulnerabilities can propagate through dependencies. A critical vulnerability in a foundational library, for example, could impact multiple models or data processing steps that depend on it.

We define the overall risk aggregation function as:
$$
\text{Overall AI System Risk} = f(\text{ComponentRisk}_1, \dots, \text{ComponentRisk}_N, \text{Interdependencies})
$$
where $f$ is an aggregation function. A simple aggregation might be the maximum risk score found in any component, or a weighted average that gives more weight to components with many downstream dependencies. For this lab, we will use a weighted sum, where components with higher 'out-degrees' (more downstream dependencies) contribute more significantly to the overall risk if their individual risk is high.

#### Code Cell (Function Implementation Description)
Implement a Python function `aggregate_overall_ai_system_risk`.
*   **Function Name:** `aggregate_overall_ai_system_risk`
*   **Parameters:** `graph: nx.Graph`
*   **Returns:** `float` (the overall AI system risk score)
*   **Details:**
    *   For each component in the graph:
        *   Get its 'Component_Risk_Score'.
        *   Get its 'out-degree' (number of direct dependencies it influences).
    *   Calculate the overall risk as the sum of (Component_Risk_Score * (1 + out_degree)) for all components.
    *   Normalize this sum to a reasonable scale, e.g., by dividing by the total number of components or a predefined maximum possible risk.

#### Code Cell (Function Execution Description)
Call `aggregate_overall_ai_system_risk` with `ai_system_graph`.
Store the result as `overall_risk_score`.
Print the `overall_risk_score` with an appropriate message.

#### Markdown Cell
By aggregating the individual component risks and considering their interdependencies, we arrive at a more holistic view of the AI system's security posture. This overall score helps in strategic decision-making regarding risk tolerance and resource allocation for mitigation efforts.

---

### Section 9: Simulating a Vulnerability

#### Markdown Cell
Simulating a vulnerability is a powerful way to understand potential impacts and identify critical risk propagation paths. We will select a component and artificially increase its 'Known_Vulnerabilities_Score' to reflect a newly discovered, severe exploit (e.g., a zero-day exploit). This simulation will then trigger a re-evaluation of risks throughout the system.

#### Code Cell (Function Implementation Description)
Implement a Python function `simulate_vulnerability_propagation`.
*   **Function Name:** `simulate_vulnerability_propagation`
*   **Parameters:** `graph: nx.Graph`, `vulnerable_component_id: str`, `base_impact_score: float`
*   **Returns:** `nx.Graph` (a copy of the graph with updated risks)
*   **Details:**
    *   Create a deep copy of the input `graph` to avoid modifying the original.
    *   Update the `Known_Vulnerabilities_Score` and recalculate `Component_Risk_Score` for the `vulnerable_component_id` using the `base_impact_score`.
    *   Propagate a attenuated impact to direct downstream dependencies: For each direct dependent, increase its `Component_Risk_Score` by a fraction of the `base_impact_score` (e.g., $0.5 \times \text{base_impact_score}$), ensuring scores do not exceed a maximum (e.g., 15).
    *   Optionally, propagate a further attenuated impact to indirect downstream dependencies (dependencies of dependencies, etc.) with a smaller fraction (e.g., $0.2 \times \text{base_impact_score}$).
    *   Ensure all component risk scores remain within a defined range (e.g., 0-15).

#### Code Cell (Function Execution Description)
Select a `vulnerable_component_id` (e.g., 'Comp_007', perhaps a 'Library' or 'Model' type component if possible).
Set `base_impact_score = 10` (representing a high-severity vulnerability).
Call `simulate_vulnerability_propagation` with `ai_system_graph`, the selected ID, and the impact score.
Store the returned graph as `simulated_risk_graph`.
Print a message indicating the simulation was performed and which component was affected.

#### Markdown Cell
By simulating a critical vulnerability, we've created a scenario to observe how such an event could escalate. This modified graph is now ready to show us the ripple effect across the entire AI system, which is invaluable for pre-emptive planning and incident response.

---

### Section 10: Visualizing Vulnerability Propagation

#### Markdown Cell
Visualizing the impact of a simulated vulnerability clearly demonstrates cascading effects. By comparing the graph before and after the simulation, we can identify which components are most affected and how risks propagate through the system's dependencies. Nodes will be color-coded based on their updated risk scores to immediately highlight highly impacted areas.

#### Code Cell (Function Implementation Description)
No new function implementation is required, we will reuse the `networkx` visualization logic.

#### Code Cell (Function Execution Description)
Visualize the `simulated_risk_graph` using `networkx` and `matplotlib.pyplot`.
*   Nodes should be colored based on their 'Component_Risk_Score' in the `simulated_risk_graph`, with a color gradient (e.g., green to red for low to high risk).
*   Optionally, node size can also be proportional to the risk score for greater emphasis.
*   Include a color bar to explain the risk score mapping.
*   Set a fixed seed for `nx.spring_layout` to maintain consistency with previous layouts.

#### Markdown Cell
This visualization offers a critical insight into the system's resilience and potential vulnerabilities. Risk Managers can clearly see which parts of the AI system are most exposed to a threat originating from a specific component, enabling them to prioritize mitigation strategies for critical paths.

---

### Section 11: Assessing Data Provenance and Integrity Risks

#### Markdown Cell
Data provenance and integrity are foundational to trustworthy AI. Compromised or poorly sourced data can lead to biased models, security vulnerabilities, and unreliable predictions. The AI-BOM helps track the origin and characteristics of data components, allowing for focused risk assessment.
Risks associated with data include:
*   **Data Poisoning**: Malicious data introduced into training sets.
*   **Data Drift**: Changes in input data distribution over time.
*   **Bias**: Unfair representation leading to discriminatory outcomes.
*   **Privacy Violations**: Sensitive information leakage.

For data components, the 'Origin' and 'Licensing_Info' attributes in our AI-BOM are particularly relevant for assessing these risks.

#### Code Cell (Function Implementation Description)
No new specific function is required, but we will filter and analyze the data component risks.

#### Code Cell (Function Execution Description)
Filter `ai_bom_df` to select only components where 'Component_Type' is 'Data'.
Display these data components, focusing on 'Component_ID', 'Origin', 'Licensing_Info', and 'Component_Risk_Score'.
Calculate and print the average 'Component_Risk_Score' specifically for 'Data' components.

#### Markdown Cell
By isolating and examining data components, we can specifically identify and address risks related to data quality, origin, and integrity. This focused analysis supports the development of strategies for robust data governance and provenance tracking.

---

### Section 12: Evaluating Third-Party Model and Component Risks

#### Markdown Cell
Third-party models, libraries, and hardware introduce external dependencies into the AI system supply chain. These components can carry inherent risks due to lack of transparency, unknown biases, or unpatched vulnerabilities. Rigorous due diligence and continuous monitoring are essential.

Risks include:
*   **Vulnerabilities in third-party code**: Unpatched CVEs in libraries.
*   **Supply chain attacks**: Malicious code injected into open-source components.
*   **Model opacity**: Difficulty in auditing pre-trained models.
*   **Licensing compliance**: Legal risks from non-compliant usage.

The 'Origin' and 'Known_Vulnerabilities_Score' attributes are particularly important here.

#### Code Cell (Function Implementation Description)
No new specific function is required, but we will filter and analyze third-party component risks.

#### Code Cell (Function Execution Description)
Filter `ai_bom_df` to select components where 'Origin' is either 'Third-Party Vendor A', 'Third-Party Vendor B', or 'Open Source Community'.
Display these third-party components, focusing on 'Component_ID', 'Component_Type', 'Origin', 'Known_Vulnerabilities_Score', and 'Component_Risk_Score'.
Calculate and print the average 'Component_Risk_Score' specifically for third-party components.

#### Markdown Cell
This targeted analysis of third-party components highlights their contribution to the overall risk profile. Understanding these external risks is vital for implementing robust vetting processes, contractual agreements, and continuous monitoring for components originating outside the organization's direct control.

---

### Section 13: Strategies for Supply Chain Transparency and Vulnerability Management

#### Markdown Cell
Effective AI risk management requires proactive strategies to enhance supply chain transparency and manage vulnerabilities. An AI-BOM is a foundational tool for these strategies, enabling:
*   **Proactive Vulnerability Scanning**: Regularly scan components listed in the AI-BOM for known CVEs.
*   **Data Provenance Verification**: Trace the origin and transformations of data components to ensure integrity.
*   **Dependency Mapping**: Understand the full dependency graph to identify critical components and propagation paths.
*   **Licensing Compliance**: Track and manage open-source and proprietary licenses to mitigate legal risks.

Implementing these strategies ensures continuous monitoring and timely response to emerging threats.

#### Code Cell (Function Implementation Description)
No specific function implementation is required for this conceptual section.

#### Code Cell (Function Execution Description)
Display the top 3 components with the highest 'Component_Risk_Score' from the *original* `ai_bom_df` (before simulation) along with their 'Component_Type' and 'Origin'.
Display the bottom 3 components with the lowest 'Component_Risk_Score'.

#### Markdown Cell
By identifying the highest and lowest risk components, we can understand where to focus our vulnerability management efforts. High-risk components, especially those from third-parties or with many dependencies, warrant immediate attention for mitigation, patching, or replacement, thereby improving the overall transparency and security of the AI supply chain.

---

### Section 14: Conclusion and Key Takeaways

#### Markdown Cell
This lab has demonstrated the fundamental utility of an AI Bill of Materials (AI-BOM) as a critical tool for AI risk management. We've explored how to:
*   **Construct and visualize an AI-BOM** to understand system dependencies.
*   **Assess individual component risks** based on attributes like vulnerabilities and origin.
*   **Aggregate these risks** to derive an overall AI system risk, considering interdependencies.
*   **Simulate vulnerability propagation** to visualize cascading impacts and identify critical paths.
*   **Analyze specific risk areas** such as data provenance and third-party components.

By applying AI-BOM principles, Risk Managers can enhance supply chain transparency, proactively identify vulnerabilities, and develop more robust strategies for securing complex AI systems.

#### Code Cell (Function Implementation Description)
No specific function implementation is required for this concluding section.

#### Code Cell (Function Execution Description)
No specific function execution is required for this concluding section.

#### Markdown Cell
The insights gained from an AI-BOM are indispensable for maintaining the security, integrity, and trustworthiness of AI deployments in an increasingly complex threat landscape. Proactive risk identification and management are key to building resilient AI systems.
