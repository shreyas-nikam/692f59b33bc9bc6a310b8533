
import streamlit as st
import pandas as pd
from application_pages import page_data_poisoning, page_adversarial_examples, page_prompt_injection

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: AI Risk Scenario Simulator")
st.divider()

st.markdown("""
This Streamlit application provides an interactive platform for **Risk Managers** to understand and simulate AI security vulnerabilities and mitigation strategies.

### Learning Goals
Upon using the application, users will be able to:
*   Identify and differentiate between various AI attack vectors: Prompt Injection, Data Poisoning, and Adversarial Examples.
*   Quantify the potential impact and likelihood of AI-related risks using a defined risk score model.
*   Evaluate the effectiveness of different mitigation and defense strategies (Data Sanitization, Adversarial Training, Safety Alignment/Input Filtering) in reducing AI risks.
*   Document simulated vulnerabilities and proposed solutions in a persistent AI Risk Register.

### Introduction to AI Risk Simulation
Artificial Intelligence (AI) systems, while powerful, introduce new security risks. Understanding and managing these risks is crucial for their safe and responsible deployment. This lab focuses on three common AI attack vectors:

1.  **Data Poisoning:** Malicious data injected into the training set to manipulate model behavior.
2.  **Adversarial Examples:** Specially crafted inputs that cause a model to misclassify, often imperceptible to humans.
3.  **Prompt Injection:** Attacks targeting Large Language Models (LLMs) to override their instructions or extract sensitive information.

**Risk Quantification:**
The core formula for risk quantification used throughout this application is:
$$
Risk = P(Event) \times M(Consequence)
$$
Where:
*   $P(Event)$ represents the probability of an attack succeeding.
*   $M(Consequence)$ denotes the magnitude of harm (e.g., financial loss, data breach severity).

Both $P$ and $M$ are qualitatively defined (e.g., 'Low', 'Medium', 'High') and mapped to numerical scales (1-5 for calculation).

**Risk Level Mapping:**
*   **Low Risk:** Score 1-5
*   **Medium Risk:** Score 6-15
*   **High Risk:** Score 16-25
""")

# Initialize risk_register_df in session state if not already present
if 'risk_register_df' not in st.session_state:
    st.session_state.risk_register_df = pd.DataFrame(columns=[
        'Attack Type', 'Description',
        'Initial P(Event)', 'Initial M(Consequence)', 'Initial Risk Score',
        'Mitigation Applied',
        'Mitigated P(Event)', 'Mitigated M(Consequence)', 'Mitigated Risk Score',
        'Performance Impact (%)', 'Performance Recovery (%)'
    ])

page = st.sidebar.selectbox(
    label="Navigation",
    options=["Data Poisoning", "Adversarial Examples", "Prompt Injection"],
    index=0 # Default to Data Poisoning
)

if page == "Data Poisoning":
    page_data_poisoning.main()
elif page == "Adversarial Examples":
    page_adversarial_examples.main()
elif page == "Prompt Injection":
    page_prompt_injection.main()

st.divider()
st.subheader("AI Risk Register")
st.markdown(r"""
This section logs all the simulated AI vulnerabilities and the effectiveness of the applied mitigation strategies.
It provides a persistent record for risk managers to review past scenarios and their outcomes.
""")

if not st.session_state.risk_register_df.empty:
    st.dataframe(st.session_state.risk_register_df, use_container_width=True)
else:
    st.info("The AI Risk Register is currently empty. Run simulations to populate it.")
