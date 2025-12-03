id: 692f59b33bc9bc6a310b8533_user_guide
summary: AI Design and Deployment Lab 6 User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# AI-BOM Risk Navigator: A Guide for AI Supply Chain Risk Management

## 1. Introduction to AI Bill of Materials (AI-BOM) for Risk Management
Duration: 00:05:00

Welcome to the **AI-BOM Risk Navigator** codelab! In this interactive guide, you will learn about the critical role of an AI Bill of Materials (AI-BOM) in managing risks across the AI supply chain. As AI systems become more complex and integrated into various business processes, understanding their constituent components and dependencies is paramount for ensuring security, trustworthiness, and compliance.

An AI-BOM provides a structured inventory of all elements that make up an AI system, including datasets, models, libraries, and hardware. This transparency is crucial for proactive risk identification and mitigation. Just as a software Bill of Materials (SBOM) tracks software components, an AI-BOM extends this concept to cover AI-specific elements like training data and model architectures. This helps in identifying potential vulnerabilities, assessing data provenance, and understanding the cascading impact of a compromised element.

**Key Learning Goals of this Codelab:**
*   Understand the fundamental concept and importance of an AI Bill of Materials (AI-BOM) for effective AI risk management.
*   Learn how to generate and interpret a synthetic AI-BOM dataset to simulate real-world scenarios.
*   Visualize AI system dependencies, gaining insights into the structural integrity and interconnections.
*   Calculate and interpret individual component risk scores, considering attributes like known vulnerabilities and component origin.
*   Aggregate overall AI System Risk, understanding how component interdependencies influence the total risk profile.
*   Simulate the propagation of a vulnerability throughout an AI system and visually assess its potential impact.
*   Perform targeted risk analysis, specifically focusing on data provenance, integrity risks, and challenges posed by third-party components.
*   Identify the highest and lowest risk components within the system to prioritize and focus risk mitigation efforts strategically.

This section lays the foundation for understanding what an AI-BOM is and why it's essential for AI risk management. Subsequent sections will build upon this by generating an AI-BOM, visualizing it, and analyzing risks.

<aside class="positive">
<b>AI-BOMs</b> are critical for establishing trust and accountability in AI systems, much like ingredient lists on food products or component lists for manufactured goods. They bring transparency to what often feels like a "black box."
</aside>

## 2. Generating a Synthetic AI-BOM Dataset and Understanding its Attributes
Duration: 00:07:00

To practically demonstrate the utility of an AI-BOM without needing real-world sensitive data, we will generate a synthetic dataset. This dataset will represent various components of an AI system and their interdependencies. Each component will have attributes crucial for risk assessment, such as type, origin, version, known vulnerabilities, and licensing information.

1.  **Navigate to the Sidebar Controls**: On the left-hand sidebar, you'll find a section titled "Configuration".

2.  **Set Number of Components and Dependencies**:
    *   Adjust the "Number of Components" slider to define how many distinct elements (e.g., datasets, models, libraries) your AI system will have. A value between 10-20 is a good starting point.
    *   Adjust the "Number of Dependencies" slider to define the number of connections between these components. This will influence the complexity of your AI system's architecture. A value between 15-30 works well for initial exploration.

3.  **Generate the AI-BOM**: Click the **"Generate AI-BOM"** button.

    <aside class="positive">
    You should see a success message: "AI-BOM Generated!" This means the application has created a synthetic dataset based on your parameters and stored it for further analysis.
    </aside>

4.  **View the AI-BOM Dataset**: Once generated, the main content area will update. You'll see a section titled "AI-BOM DataFrame" displaying the tabular data of your generated components.

    The generated synthetic AI-BOM data provides a realistic foundation for our analysis. We can now see the various components that constitute our hypothetical AI system and their initial risk-relevant attributes. This table is the raw input for building our dependency graph.

5.  **Understand Component Attributes**: Each entry in the AI-BOM describes a component with several key attributes. These attributes are critical for assessing individual component risks and their potential impact on the overall AI system.
    *   **Component Name/ID**: A unique identifier for the component.
    *   **Component Type**: Categorizes the component (e.g., 'Data', 'Model', 'Library', 'Hardware'). Different types may have different risk profiles.
    *   **Origin/Provenance**: Where the component came from (e.g., 'Internal', 'Third-Party Vendor A', 'Open Source Community'). Crucial for assessing supply chain risks.
    *   **Version**: Specific version details, important for tracking known vulnerabilities.
    *   **Known Vulnerabilities (Score)**: A numerical representation of identified security weaknesses. We use a simplified score, conceptually related to CVSS.
    *   **Licensing Information**: Details about the license, which can imply legal or security risks.

    These attributes, especially 'Known_Vulnerabilities_Score' and 'Origin', will directly influence a component's risk profile.

    The application also provides "AI-BOM DataFrame Information" (similar to a `df.info()` output) and "AI-BOM DataFrame Descriptive Statistics" to give you a quick overview of the data types and statistical distribution of the attributes.

    By examining the `ai_bom_df`'s attributes, we get a clearer picture of the data we're working with. Understanding these attributes is the first step in identifying potential risk factors associated with each component in our AI system.

## 3. Visualizing AI System Dependencies (Initial Graph)
Duration: 00:06:00

Understanding the intricate web of dependencies within an AI system is crucial for identifying critical paths and potential single points of failure. A visual representation makes this complexity much easier to grasp for Risk Managers.

1.  **Navigate to "Initial Dependencies"**: In the main navigation dropdown (or sidebar if present), select the "Initial Dependencies" page.

2.  **Observe the Dependency Graph**: You will see a graph where:
    *   **Nodes** represent individual components from your AI-BOM (e.g., `Comp_001`, `Model A`).
    *   **Edges (arrows)** represent dependencies, showing how one component relies on another. For example, an arrow from 'Data_001' to 'Model_002' means 'Model_002' depends on 'Data_001'.
    *   **Colors** indicate the 'Component Type' (Data, Model, Library, Hardware), with a legend provided for easy identification.
    *   **Labels** display the component ID for clear identification.

    <aside class="negative">
    If you don't see a graph, ensure you have first generated the AI-BOM dataset in the previous step using the sidebar controls.
    </aside>

    The visual graph provides an immediate understanding of the AI system's architecture. We can see how different components are interconnected, which is fundamental for identifying critical paths and potential single points of failure. The color-coding by component type gives a quick overview of the system's composition. This visualization is invaluable for a Risk Manager to quickly assess the system's structural integrity and identify areas of high connectivity which could imply higher risk propagation.

## 4. Introduction to Vulnerability Scoring
Duration: 00:05:00

Now that we have our components and their dependencies, let's delve into how we quantify risk. A key part of risk assessment for any component is understanding its vulnerabilities.

1.  **Navigate to "Component Risk Profile"**: From the navigation, select "Component Risk Profile".

2.  **Understand Vulnerability Scoring**: This section introduces the concept of vulnerability scoring. Vulnerability scoring provides a standardized way to quantify the severity of security weaknesses. The Common Vulnerability Scoring System (CVSS) is a widely used open framework for communicating these characteristics and impacts. A simplified version of CVSS helps in prioritizing risks.

    The CVSS score is derived from various metrics, broadly categorized into base, temporal, and environmental metrics. For our simplified scenario, we can represent this as:
    $$ \text{CVSS Score} = g(\text{AttackVector}, \text{AttackComplexity}, \text{PrivilegesRequired}, \dots) $$
    where $g$ is a function that combines several factors related to the vulnerability's exploitability and impact. In our synthetic data, 'Known_Vulnerabilities_Score' directly serves as this simplified score. A higher score indicates a more severe vulnerability.

    The application displays key metrics for `Known_Vulnerabilities_Score`:
    *   **Maximum Known Vulnerabilities Score**
    *   **Minimum Known Vulnerabilities Score**
    *   **Average Known Vulnerabilities Score**

    These metrics give you a quick baseline of the security posture across all components in your AI-BOM. Understanding the range and distribution of vulnerability scores in our AI-BOM gives us a baseline for assessing the security posture of individual components. This numerical value is a direct input for our component risk calculations.

## 5. Calculating Individual Component Risk
Duration: 00:05:00

Each component in the AI system carries its own set of risks, influenced by attributes like its type, origin, and known vulnerabilities. To assess these individual risks, we combine these attributes into a single 'Component Risk' score.

1.  **Review the Risk Calculation Concept**: As explained in the application, a simple risk function considers the 'Known_Vulnerabilities_Score' and potentially the 'Origin' (e.g., third-party components might inherently carry higher risk). For instance, third-party vendor components might add a penalty of 2.0 to the base vulnerability score, while open-source components might add 1.0. The score is capped at 15.0 for consistency.

2.  **View Component Risk Scores**: The application immediately shows the "First 5 components with their vulnerability and calculated risk scores". This table provides a direct view of how the `Known_Vulnerabilities_Score` and `Origin` contribute to the `Component_Risk_Score` for each component.

    We have now quantified the individual risk associated with each component. This 'Component_Risk_Score' provides a granular view of where potential problems might lie within the AI system, taking into account both reported vulnerabilities and supply chain factors.

## 6. Displaying Component Risk Profiles
Duration: 00:03:00

Beyond just a score, a human-readable risk profile offers a concise summary of a component's key risk-relevant information. This allows Risk Managers to quickly understand the implications of a specific component's presence in the AI system without needing to parse raw data.

1.  **Select a Component**: Use the "Select a component to view its Risk Profile" dropdown. Choose any component ID from the list.

2.  **Read the Risk Profile**: A textual summary will appear, detailing:
    *   Component Type
    *   Origin
    *   Version
    *   Known Vulnerabilities Score
    *   Licensing Info
    *   Calculated Component Risk Score

    The component risk profile offers a quick and comprehensive summary, allowing Risk Managers to efficiently assess specific components. This is crucial for focused investigations and for understanding the details behind a component's assigned risk score.

## 7. Aggregating Overall AI System Risk
Duration: 00:04:00

The overall AI System Risk is not just the sum of individual component risks. It must also account for how vulnerabilities can propagate through dependencies. A critical vulnerability in a foundational library, for example, could impact multiple models or data processing steps that depend on it.

1.  **Understand Overall Risk Aggregation**: The application explains that the overall risk aggregation function considers individual component risks and their interdependencies. A simple aggregation might be the maximum risk score, or a weighted average that gives more weight to components with many downstream dependencies. For this lab, a weighted sum is used, where components with higher 'out-degrees' (more downstream dependencies) contribute more significantly to the overall risk if their individual risk is high.

    The formula for the overall AI System Risk can be conceptualized as:
    $$ \text{Overall AI System Risk} = f(\text{ComponentRisk}_1, \dots, \text{ComponentRisk}_N, \text{Interdependencies}) $$

2.  **View the Overall AI System Risk Score**: A clear metric labeled "Overall Calculated AI System Risk Score (0-100)" is displayed. This score provides a high-level summary of your entire AI system's risk posture.

    By aggregating the individual component risks and considering their interdependencies, we arrive at a more holistic view of the AI system's security posture. This overall score helps in strategic decision-making regarding risk tolerance and resource allocation for mitigation efforts.

## 8. Simulating a Vulnerability
Duration: 00:05:00

Simulating a vulnerability is a powerful way to understand potential impacts and identify critical risk propagation paths. We will select a component and artificially increase its 'Known_Vulnerabilities_Score' to reflect a newly discovered, severe exploit (e.g., a zero-day exploit). This simulation will then trigger a re-evaluation of risks throughout the system.

1.  **Navigate to the Sidebar Controls**: Go back to the left-hand sidebar.

2.  **Select a Vulnerable Component**: Under the "2. Simulate Vulnerability" section, use the "Select Vulnerable Component ID" dropdown to choose a component that you want to simulate a vulnerability for.

3.  **Set the Base Impact Score**: Adjust the "Base Impact Score" slider. This value represents how severely the chosen component's vulnerability score will increase. A higher score simulates a more critical vulnerability.

4.  **Run the Simulation**: Click the **"Run Vulnerability Simulation"** button.

    <aside class="positive">
    You should see a success message indicating the simulation is complete. This action triggers a recalculation of risk scores, propagating the impact through dependent components.
    </aside>

    By simulating a critical vulnerability, we've created a scenario to observe how such an event could escalate. This modified graph is now ready to show us the ripple effect across the entire AI system, which is invaluable for pre-emptive planning and incident response.

## 9. Visualizing Vulnerability Propagation
Duration: 00:06:00

After running the simulation, it's time to see the impact. Visualizing the effect of a simulated vulnerability clearly demonstrates cascading effects. By comparing the graph before and after the simulation, we can identify which components are most affected and how risks propagate through the system's dependencies.

1.  **Navigate to "Vulnerability Impact"**: From the navigation, select "Vulnerability Impact".

2.  **Observe the Impacted Graph**: You will see a new dependency graph. Notice the changes:
    *   **Node Colors**: Nodes are now color-coded based on their updated risk scores. A color bar on the right will show the mapping from color to 'Component Risk Score' (typically from green for low risk to red for high risk).
    *   **Node Sizes**: Nodes will also vary in size, with larger nodes indicating a higher `Component_Risk_Score`. This visual cue helps you immediately spot the most affected components.
    *   **Propagated Impact**: The directly vulnerable component will show a significantly increased risk. Its direct and indirect downstream dependencies will also show increased risk, albeit attenuated (less severe) the further away they are.

    This visualization offers a critical insight into the system's resilience and potential vulnerabilities. Risk Managers can clearly see which parts of the AI system are most exposed to a threat originating from a specific component, enabling them to prioritize mitigation strategies for critical paths. This proactive understanding is essential for building robust and secure AI systems.

## 10. Assessing Data Provenance and Integrity Risks
Duration: 00:05:00

Beyond general component risks, specific risks relate to the core elements of AI: data. Data provenance and integrity are foundational to trustworthy AI. Compromised or poorly sourced data can lead to biased models, security vulnerabilities, and unreliable predictions. The AI-BOM helps track the origin and characteristics of data components, allowing for focused risk assessment.

1.  **Navigate to "Targeted Risk Analysis"**: From the navigation, select "Targeted Risk Analysis".

2.  **Understand Data Risks**: The section outlines typical risks associated with data, such as Data Poisoning, Data Drift, Bias, and Privacy Violations. For data components, the 'Origin' and 'Licensing_Info' attributes in our AI-BOM are particularly relevant for assessing these risks.

3.  **Review Data Components**: The application filters and displays "Data Components and their Risk-Relevant Attributes", showing their IDs, Origin, Licensing Info, and calculated Risk Score. An average risk score for data components is also provided.

    By isolating and examining data components, we can specifically identify and address risks related to data quality, origin, and integrity. This focused analysis supports the development of strategies for robust data governance and provenance tracking.

## 11. Evaluating Third-Party Model and Component Risks
Duration: 00:04:00

Modern AI systems heavily rely on third-party models, libraries, and hardware. While beneficial for development speed, these external dependencies introduce unique supply chain risks due to potential lack of transparency, unknown biases, or unpatched vulnerabilities. Rigorous due diligence and continuous monitoring are essential.

1.  **Understand Third-Party Risks**: This section highlights specific risks, including vulnerabilities in third-party code, supply chain attacks, model opacity, and licensing compliance issues. The 'Origin' and 'Known_Vulnerabilities_Score' attributes are particularly important here.

2.  **Review Third-Party Components**: The application filters and displays "Third-Party Components and their Risk-Relevant Attributes", including Component Type, Origin, Known Vulnerabilities Score, and Component Risk Score. An average risk score for third-party components is calculated.

    This targeted analysis of third-party components highlights their contribution to the overall risk profile. Understanding these external risks is vital for implementing robust vetting processes, contractual agreements, and continuous monitoring for components originating outside the organization's direct control.

## 12. Identifying Top and Bottom Risk Components
Duration: 00:03:00

To effectively prioritize mitigation efforts, Risk Managers need to quickly identify which components pose the highest threat and which are relatively secure.

1.  **Set 'N' for Top/Bottom Components**: You'll see an input field labeled "Enter N for Top/Bottom Components". Adjust this number to specify how many of the highest and lowest risk components you want to view. For example, enter `3` to see the top 3 and bottom 3 components.

2.  **Review Top and Bottom Components**:
    *   The "Top N components with the highest risk scores" table will show components that require immediate attention.
    *   The "Bottom N components with the lowest risk scores" table helps identify relatively stable parts of the system.

    By identifying the highest and lowest risk components, we can understand where to focus our vulnerability management efforts. High-risk components, especially those from third-parties or with many dependencies, warrant immediate attention for mitigation, patching, or replacement, thereby improving the overall transparency and security of the AI supply chain.

The insights gained from an AI-BOM are indispensable for maintaining the security, integrity, and trustworthiness of AI deployments in an increasingly complex threat landscape. Proactive risk identification and management are key to building resilient AI systems.
