Here's a comprehensive `README.md` file for your Streamlit application lab project.

---

# QuLab: AI-BOM Risk Navigator

## Project Title

**AI-BOM Risk Navigator: Interactive AI Supply Chain Risk Management**

## Project Description

The **AI-BOM Risk Navigator** is an interactive Streamlit application developed as part of the QuLab initiative. It provides a powerful, visual, and data-driven platform for Risk Managers to explore and assess risks within complex AI system supply chains. Leveraging the concept of an AI Bill of Materials (AI-BOM), the application demonstrates how to inventory AI components, understand their interdependencies, identify vulnerabilities, and simulate the cascading impact of compromised elements.

This lab project aims to demystify AI risk management by offering a hands-on experience in generating, visualizing, and analyzing synthetic AI-BOM data. Users can configure AI systems, calculate component and system-wide risks, and run simulations to understand vulnerability propagation, making it an invaluable tool for learning and practical application in AI governance and security.

## Features

The AI-BOM Risk Navigator offers the following key functionalities:

*   **Introduction to AI-BOM**: A foundational overview of what an AI Bill of Materials is and its critical role in AI risk management.
*   **Synthetic AI-BOM Generation**: Dynamically generate a synthetic AI-BOM dataset with configurable numbers of components and dependencies, featuring various component types, origins, versions, vulnerabilities, and licensing info.
*   **AI-BOM Data Details**: View the generated AI-BOM as a DataFrame, inspect its structure, data types, and descriptive statistics.
*   **Initial Dependency Visualization**: Visualize the AI system's architecture as a directed graph, showing components (nodes) and their interdependencies (edges), color-coded by component type.
*   **Vulnerability Scoring Explanation**: Understand the concept of vulnerability scoring (simplified CVSS-like scores) and its application in AI-BOM.
*   **Individual Component Risk Calculation**: Calculate and display a risk score for each AI component based on attributes like known vulnerabilities and origin (e.g., third-party).
*   **Component Risk Profiles**: Generate human-readable risk profiles for selected components, summarizing their risk-relevant attributes and calculated scores.
*   **Overall AI System Risk Aggregation**: Compute an aggregated overall risk score for the entire AI system, considering both individual component risks and their interdependencies (weighted by out-degree).
*   **Vulnerability Simulation**: Simulate a critical vulnerability in a chosen component and observe how its impact propagates through direct and indirect dependencies.
*   **Vulnerability Propagation Visualization**: Visually demonstrate the cascading effects of a simulated vulnerability, with nodes re-colored and sized based on their updated risk scores.
*   **Targeted Risk Analysis**:
    *   Assess **Data Provenance and Integrity Risks** by filtering and analyzing 'Data' type components.
    *   Evaluate **Third-Party Component Risks** by focusing on components from external vendors or open-source origins.
    *   Identify **Top and Bottom N Risk Components** to prioritize mitigation efforts.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/ai-bom-risk-navigator.git
    cd ai-bom-risk-navigator
    ```

    *(Note: Replace `https://github.com/your-username/ai-bom-risk-navigator.git` with the actual repository URL.)*

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file in the root directory of the project with the following content:

    ```
    streamlit>=1.0.0
    pandas>=1.3.0
    numpy>=1.21.0
    networkx>=2.6.0
    matplotlib>=3.4.0
    seaborn>=0.11.0
    ```

    Then install:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  Your web browser should automatically open to the Streamlit application (usually `http://localhost:8501`).

3.  **Explore the Application:**
    *   **Sidebar Controls**:
        *   **Generate AI-BOM Dataset**: Use the sliders in the sidebar to configure the "Number of Components" and "Number of Dependencies". Click "Generate AI-BOM" to create a synthetic dataset and initialize the AI system graph. This is the first step to interact with the application.
        *   **Simulate Vulnerability**: Once an AI-BOM is generated, select a "Vulnerable Component ID" from the dropdown and set a "Base Impact Score". Click "Run Vulnerability Simulation" to see the propagation of risk.
    *   **Navigation**: Use the "Navigation" selectbox in the sidebar to switch between different sections of the lab project:
        *   **Introduction**: Learn about AI-BOMs.
        *   **AI-BOM Details**: View the generated dataset and component attributes.
        *   **Initial Dependencies**: See the initial graph visualization of the AI system.
        *   **Component Risk Profile**: Explore vulnerability scoring, individual component risks, and overall system risk.
        *   **Vulnerability Impact**: Observe the visualization of a simulated vulnerability's propagation.
        *   **Targeted Risk Analysis**: Conduct specific analyses on data provenance, third-party components, and identify top/bottom risk components.

The application leverages Streamlit's `st.session_state` to maintain the generated AI-BOM data and graph across different pages and interactions, ensuring a consistent user experience.

## Project Structure

```
.
├── application_pages/
│   ├── page_1_introduction.py            # Introduction to AI-BOM concept
│   ├── page_2_ai_bom_details.py          # Synthetic AI-BOM generation and data display
│   ├── page_3_initial_dependencies.py    # Initial AI system dependency visualization
│   ├── page_4_component_risk_profile.py  # Vulnerability scoring, component & overall risk calculation
│   ├── page_5_vulnerability_impact.py    # Vulnerability simulation and impact visualization
│   └── page_6_targeted_risk_analysis.py  # Data provenance, third-party, and top/bottom N risk analysis
├── app.py                                # Main Streamlit application file, orchestrates pages and functions
├── requirements.txt                      # List of Python dependencies
└── README.md                             # Project README file
```

*   `app.py`: This is the main entry point for the Streamlit application. It handles session state initialization, sidebar controls for global actions (generate AI-BOM, simulate vulnerability), and navigates between different content pages. It also defines core functions like `generate_ai_bom_dataset`, `create_ai_system_graph`, `calculate_component_risk`, `aggregate_overall_ai_system_risk`, and `simulate_vulnerability_propagation`.
*   `application_pages/`: This directory contains individual Python modules, each representing a distinct page or section of the Streamlit application. This modular structure helps organize the content and logic for different features.

## Technology Stack

*   **Python 3.8+**: The core programming language.
*   **Streamlit**: The open-source app framework used to build and deploy interactive web applications for machine learning and data science.
*   **Pandas**: For data manipulation and analysis, primarily for managing the AI-BOM dataset (`ai_bom_df`).
*   **NumPy**: Essential for numerical operations, often used implicitly by Pandas.
*   **NetworkX**: A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks (graphs). Used for representing AI system dependencies.
*   **Matplotlib**: A comprehensive library for creating static, animated, and interactive visualizations in Python.
*   **Seaborn**: A Python data visualization library based on matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.

## Contributing

Contributions to the AI-BOM Risk Navigator are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Commit your changes (`git commit -m 'Add new feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

Please ensure your code adheres to good practices and includes appropriate documentation and tests where applicable.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You might need to create a `LICENSE` file in your repository with the MIT license text.)*

## Contact

For questions, feedback, or support related to this project, please contact:

*   **Organization**: QuantUniversity (QuLab)
*   **Email**: info@quantuniversity.com
*   **Website**: [https://www.quantuniversity.com](https://www.quantuniversity.com)

---