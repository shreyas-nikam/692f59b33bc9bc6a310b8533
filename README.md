# QuLab: AI Risk Scenario Simulator

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: AI Risk Scenario Simulator** is an interactive Streamlit application designed as an educational and simulation platform for **Risk Managers** and AI practitioners. It aims to demystify AI security vulnerabilities and demonstrate the effectiveness of various mitigation strategies in an approachable, hands-on manner.

The application allows users to simulate three prevalent AI attack vectors: Data Poisoning, Adversarial Examples, and Prompt Injection. For each scenario, users can observe the attack's impact on AI model performance or safety, quantify the associated risks using a simple yet effective risk scoring model, and then apply and evaluate mitigation techniques. All simulated scenarios and their outcomes are logged in a persistent AI Risk Register for comprehensive review and analysis.

### Learning Goals

Upon using this application, users will be able to:

*   **Identify and Differentiate**: Understand the distinct characteristics and mechanisms of Data Poisoning, Adversarial Examples, and Prompt Injection attacks.
*   **Quantify Risk**: Apply a defined risk score model to quantify the potential impact and likelihood of AI-related risks.
*   **Evaluate Mitigations**: Assess the effectiveness of common defense strategies (e.g., Data Sanitization, Adversarial Training, Safety Alignment/Input Filtering) in reducing identified AI risks.
*   **Document and Analyze**: Utilize a persistent AI Risk Register to document simulated vulnerabilities, proposed solutions, and their performance/safety implications.

### Introduction to AI Risk Simulation

Artificial Intelligence (AI) systems, while revolutionary, introduce unique security risks that must be understood and managed for their safe and responsible deployment. This lab project focuses on demonstrating and analyzing three critical AI attack vectors:

1.  **Data Poisoning**: Involves injecting malicious data into a model's training dataset to manipulate its behavior, leading to incorrect associations or degraded performance.
2.  **Adversarial Examples**: Specially crafted inputs (often imperceptible to humans) designed to cause an AI model to make a wrong classification or prediction.
3.  **Prompt Injection**: Attacks specifically targeting Large Language Models (LLMs) by crafting inputs that override system instructions, bypass safety guidelines, or extract sensitive information.

**Risk Quantification Model:**
The core formula for risk quantification used throughout this application is:
$$
Risk = P(Event) \times M(Consequence)
$$
Where:
*   $P(Event)$ represents the probability of an attack succeeding.
*   $M(Consequence)$ denotes the magnitude of harm (e.g., financial loss, data breach severity).

Both $P$ and $M$ are qualitatively defined (e.g., 'Low', 'Medium', 'High') and mapped to numerical scales (1-5 for calculation).

**Risk Level Mapping:**
*   **Low Risk**: Score 1-5
*   **Medium Risk**: Score 6-15
*   **High Risk**: Score 16-25

## Features

*   **Interactive Attack Simulations**:
    *   **Data Poisoning**: Simulate poisoning a CNN's training data to misclassify images, with adjustable poisoning rates.
    *   **Adversarial Examples**: Generate adversarial images for a CNN using adjustable epsilon values, demonstrating misclassification with imperceptible changes.
    *   **Prompt Injection**: Test a simplified LLM against safe and malicious prompts, including custom user inputs, to expose vulnerabilities.
*   **Dynamic Risk Assessment**: Qualitatively define and numerically calculate initial, post-attack, and mitigated risk scores based on $P(Event)$ and $M(Consequence)$.
*   **Mitigation Strategy Demonstration**:
    *   **Data Sanitization**: Apply simulated data cleaning to counter data poisoning attacks.
    *   **Adversarial Training**: Re-train a model with adversarial examples to enhance robustness.
    *   **Safety Alignment/Input Filtering**: Implement keyword-based filtering to prevent prompt injection in LLMs.
*   **Performance Visualization**: Visualize the impact of attacks and effectiveness of mitigations using interactive bar charts for model accuracy or safety scores.
*   **Persistent AI Risk Register**: Automatically logs details of each simulated scenario, including attack type, description, risk scores (initial, post-attack, mitigated), mitigation applied, and performance impact/recovery metrics.
*   **User-Friendly Interface**: Built with Streamlit for an intuitive and responsive web application experience.

## Getting Started

Follow these instructions to set up and run the QuLab application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/quolab-ai-risk-simulator.git
    cd quolab-ai-risk-simulator
    ```

    *(Note: Replace `your-username/quolab-ai-risk-simulator.git` with the actual repository URL if this is hosted.)*

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The application relies on several Python libraries. It's recommended to install them using a `requirements.txt` file.

    First, create a `requirements.txt` file in the root directory of the project with the following content:

    ```
    streamlit==1.32.2
    numpy==1.26.4
    pandas==2.2.1
    matplotlib==3.8.3
    seaborn==0.13.2
    scikit-learn==1.4.1.post1
    torch==2.2.1
    torchvision==0.17.1
    scikit-image==0.22.0
    ```
    *(Note: These versions are specified for reproducibility. You might use slightly newer versions, but compatibility should be checked.)*

    Then, install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**

    Navigate to the project's root directory in your terminal (if you're not already there) and run:

    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**
    Your web browser will automatically open to `http://localhost:8501` (or another port if 8501 is in use), displaying the QuLab application.

3.  **Navigate Scenarios:**
    Use the sidebar to switch between different AI attack vector simulations: "Data Poisoning", "Adversarial Examples", and "Prompt Injection".

4.  **Run Simulations:**
    *   For each scenario, observe the baseline performance of the AI system.
    *   Adjust attack parameters (e.g., "Poisoning Rate", "Epsilon", custom malicious prompts).
    *   Set initial qualitative risk levels ($P(Event)$, $M(Consequence)$).
    *   Click the "Run [Attack Type] Scenario" button to execute the attack simulation and observe its impact on performance/safety and the updated risk score.
    *   (Optional) Adjust mitigation parameters (e.g., "Detection Threshold").
    *   Set mitigated qualitative risk levels.
    *   Click "Apply [Mitigation Strategy]" to see how the defenses improve the system and reduce risk.

5.  **Review Risk Register:**
    Scroll to the bottom of the main page to view the "AI Risk Register", which logs all your simulated scenarios and their outcomes.

## Project Structure

```
.
├── application_pages/
│   ├── page_data_poisoning.py         # Streamlit page for Data Poisoning simulation
│   ├── page_adversarial_examples.py   # Streamlit page for Adversarial Examples simulation
│   └── page_prompt_injection.py       # Streamlit page for Prompt Injection simulation
├── app.py                             # Main Streamlit application, navigation, and risk register
├── utils.py                           # Core utility functions (risk calcs, model defs, attack/mitigation logic)
└── requirements.txt                   # List of Python dependencies
```

## Technology Stack

*   **Streamlit**: For creating interactive web applications with Python.
*   **Python**: The core programming language.
*   **NumPy**: For numerical operations, especially with array manipulation for image data.
*   **Pandas**: For data handling and managing the AI Risk Register.
*   **Matplotlib & Seaborn**: For generating data visualizations and performance charts.
*   **scikit-learn**: For data splitting (`train_test_split`).
*   **PyTorch**: For building, training, and evaluating the Convolutional Neural Network (CNN) models and generating adversarial examples.
*   **scikit-image (skimage.draw)**: For generating synthetic image shapes (circles, rectangles, triangles).

## Contributing

Contributions are welcome! If you have suggestions for new features, improvements, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
*(Note: You will need to create a `LICENSE` file in your repository if you haven't already.)*

## Contact

For any questions or feedback, please reach out:

*   **Project Maintainer**: Your Name / Organization Name
*   **Email**: your.email@example.com
*   **Website**: [https://www.quantuniversity.com](https://www.quantuniversity.com) (as seen in the logo)
