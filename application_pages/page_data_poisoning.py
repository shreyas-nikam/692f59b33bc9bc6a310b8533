
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils import (
    generate_synthetic_image_data, define_simple_cnn_model, train_model,
    evaluate_model_performance, simulate_data_poisoning,
    apply_mitigation_data_sanitization, calculate_risk_score,
    display_risk_status, create_performance_chart, add_to_risk_register,
    num_samples, img_size, epochs, lr, class_names
)

@st.cache_data(ttl="2h")
def get_base_data():
    images, labels = generate_synthetic_image_data(num_samples, img_size)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return X_train, y_train, X_test, y_test, train_loader, test_loader

@st.cache_resource
def get_and_train_baseline_model(train_loader):
    model = define_simple_cnn_model()
    train_model(model, train_loader, epochs, lr)
    return model

def main():
    st.header("Understanding AI Attack Vector 1: Data Poisoning")
    st.markdown(r"""
    Data poisoning attacks involve injecting malicious data into a model's training dataset.
    This can manipulate the model's behavior, making it learn incorrect associations or
    perform poorly on specific tasks.

    In this simulation, we'll observe how a data poisoning attack can degrade the performance
    of an image classifier, specifically by making it misclassify circles as triangles.

    **Risk Formula:**
    $$
    Risk = P(Event) \times M(Consequence)
    $$
    Where $P(Event)$ is the probability of the attack succeeding, and $M(Consequence)$ is the
    magnitude of harm. Both are rated qualitatively (Low, Medium, High) and mapped to
    numerical values (1-5). A higher score indicates higher risk.
    """)

    if 'risk_register_df' not in st.session_state:
        st.session_state.risk_register_df = pd.DataFrame(columns=[
            'Attack Type', 'Description',
            'Initial P(Event)', 'Initial M(Consequence)', 'Initial Risk Score',
            'Mitigation Applied',
            'Mitigated P(Event)', 'Mitigated M(Consequence)', 'Mitigated Risk Score',
            'Performance Impact (%)', 'Performance Recovery (%)'
        ])

    st.subheader("Simulating a Baseline AI System: Image Classifier")
    st.markdown(r"""
    Our baseline system is a simple Convolutional Neural Network (CNN) trained to classify
    synthetic images of circles, squares, and triangles. We'll first evaluate its
    performance under normal conditions.
    """)

    X_train, y_train, X_test, y_test, train_loader, test_loader = get_base_data()
    baseline_model = get_and_train_baseline_model(train_loader)
    baseline_accuracy = evaluate_model_performance(baseline_model, test_loader)

    st.markdown(f"**Baseline Model Accuracy:** `{baseline_accuracy:.2f}%`")
    st.markdown(r"""
    *Annotation:* This metric represents the percentage of correctly classified images by the model.
    """)

    st.subheader("Data Poisoning Attack Scenario")
    st.markdown(r"""
    We simulate a data poisoning attack by altering a portion of the training data.
    Specifically, we will poison data points labeled as 'Circle' (class 0) to be mislabeled
    as 'Triangle' (class 2).
    """)

    poison_rate = st.slider("Select Poisoning Rate (%)", 0, 50, 10) / 100.0
    initial_p_event_dp = st.selectbox(
        "Initial P(Event) - Data Poisoning",
        options=["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index("Medium"),
        key="dp_p_event_initial"
    )
    initial_m_consequence_dp = st.selectbox(
        "Initial M(Consequence) - Data Poisoning",
        options=["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index("High"),
        key="dp_m_consequence_initial"
    )

    if st.button("Run Data Poisoning Scenario", key="run_dp_scenario"):
        with st.spinner("Running data poisoning simulation..."):
            # Simulate poisoning
            poisoned_X_train, poisoned_y_train = simulate_data_poisoning(
                X_train, y_train, poison_rate=poison_rate, target_class=0, poison_class=2
            )
            poisoned_train_dataset = TensorDataset(
                torch.tensor(poisoned_X_train, dtype=torch.float32),
                torch.tensor(poisoned_y_train, dtype=torch.long)
            )
            poisoned_train_loader = DataLoader(poisoned_train_dataset, batch_size=32, shuffle=True)

            # Re-train model on poisoned data
            poisoned_model = define_simple_cnn_model()
            train_model(poisoned_model, poisoned_train_loader, epochs, lr)
            poisoned_accuracy = evaluate_model_performance(poisoned_model, test_loader)

            st.session_state["dp_poisoned_accuracy"] = poisoned_accuracy

            st.subheader("Results After Data Poisoning Attack")
            st.markdown(f"**Poisoned Model Accuracy:** `{poisoned_accuracy:.2f}%`")

            performance_impact_dp = baseline_accuracy - poisoned_accuracy
            st.markdown(r"""
            *Annotation:* **Performance Impact (%)** measures the percentage decrease in model accuracy
            due to the attack: $$ \text{Impact} = \text{Baseline Accuracy} - \text{Attacked Accuracy} $$.
            """)
            st.markdown(f"**Performance Impact:** `{performance_impact_dp:.2f}%`")

            post_attack_p_event_dp = "High" if performance_impact_dp > 10 else "Medium"
            post_attack_m_consequence_dp = "High" if performance_impact_dp > 10 else "Medium"
            post_attack_risk_score_dp = calculate_risk_score(post_attack_p_event_dp, post_attack_m_consequence_dp)

            st.markdown(display_risk_status(post_attack_risk_score_dp), unsafe_allow_html=True)

            metrics_dp = {
                "Baseline Accuracy": baseline_accuracy,
                "Poisoned Accuracy": poisoned_accuracy
            }
            st.pyplot(create_performance_chart(metrics_dp, "Model Accuracy: Baseline vs. Poisoned"))

            st.session_state["dp_post_attack_p_event"] = post_attack_p_event_dp
            st.session_state["dp_post_attack_m_consequence"] = post_attack_m_consequence_dp
            st.session_state["dp_performance_impact"] = performance_impact_dp
            st.session_state["dp_poisoned_model"] = poisoned_model # Store for mitigation
            st.session_state["dp_poisoned_X_train"] = poisoned_X_train
            st.session_state["dp_poisoned_y_train"] = poisoned_y_train
            st.session_state["dp_original_y_train"] = y_train

            st.success("Data poisoning simulation complete!")

    if "dp_poisoned_accuracy" in st.session_state:
        st.subheader("Mitigation Strategy 1: Data Sanitization (for Poisoned Data)")
        st.markdown(r"""
        Data sanitization aims to detect and remove malicious data points from the training set.
        In this simulated scenario, we'll apply a simple detection mechanism and revert a portion
        of the poisoned labels back to their original state.
        """)
        detection_threshold = st.slider("Select Detection Threshold (%)", 0, 100, 50) / 100.0
        mitigated_p_event_dp = st.selectbox(
            "Mitigated P(Event) - Data Poisoning",
            options=["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index("Low"),
            key="dp_p_event_mitigated"
        )
        mitigated_m_consequence_dp = st.selectbox(
            "Mitigated M(Consequence) - Data Poisoning",
            options=["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index("Low"),
            key="dp_m_consequence_mitigated"
        )

        if st.button("Apply Data Sanitization", key="apply_dp_mitigation"):
            with st.spinner("Applying data sanitization..."):
                sanitized_X_train, sanitized_y_train = apply_mitigation_data_sanitization(
                    st.session_state["dp_poisoned_X_train"],
                    st.session_state["dp_poisoned_y_train"],
                    st.session_state["dp_original_y_train"],
                    detection_threshold=detection_threshold
                )
                sanitized_train_dataset = TensorDataset(
                    torch.tensor(sanitized_X_train, dtype=torch.float32),
                    torch.tensor(sanitized_y_train, dtype=torch.long)
                )
                sanitized_train_loader = DataLoader(sanitized_train_dataset, batch_size=32, shuffle=True)

                mitigated_model_dp = define_simple_cnn_model()
                train_model(mitigated_model_dp, sanitized_train_loader, epochs, lr)
                mitigated_accuracy_dp = evaluate_model_performance(mitigated_model_dp, test_loader)

                st.session_state["dp_mitigated_accuracy"] = mitigated_accuracy_dp

                st.subheader("Results After Data Sanitization")
                st.markdown(f"**Mitigated Model Accuracy:** `{mitigated_accuracy_dp:.2f}%`")

                performance_recovery_dp = mitigated_accuracy_dp - st.session_state["dp_poisoned_accuracy"]
                st.markdown(r"""
                *Annotation:* **Performance Recovery (%)** indicates the improvement in model accuracy
                after applying mitigation: $$ \text{Recovery} = \text{Mitigated Accuracy} - \text{Attacked Accuracy} $$.
                """)
                st.markdown(f"**Performance Recovery:** `{performance_recovery_dp:.2f}%`")

                mitigated_risk_score_dp = calculate_risk_score(mitigated_p_event_dp, mitigated_m_consequence_dp)
                st.markdown(display_risk_status(mitigated_risk_score_dp), unsafe_allow_html=True)

                metrics_mitigated_dp = {
                    "Baseline Accuracy": baseline_accuracy,
                    "Poisoned Accuracy": st.session_state["dp_poisoned_accuracy"],
                    "Mitigated Accuracy": mitigated_accuracy_dp
                }
                st.pyplot(create_performance_chart(metrics_mitigated_dp, "Model Accuracy: Baseline, Poisoned, Mitigated"))

                add_to_risk_register(
                    attack_type="Data Poisoning",
                    description=f"Targeting Circle (class 0) to be misclassified as Triangle (class 2) with {int(poison_rate*100)}% poison rate.",
                    initial_p_qual=initial_p_event_dp,
                    initial_m_qual=initial_m_consequence_dp,
                    post_attack_p_qual=st.session_state["dp_post_attack_p_event"],
                    post_attack_m_qual=st.session_state["dp_post_attack_m_consequence"],
                    mitigation_applied="Data Sanitization",
                    mitigated_p_qual=mitigated_p_event_dp,
                    mitigated_m_qual=mitigated_m_consequence_dp,
                    perf_impact_perc=st.session_state["dp_performance_impact"],
                    perf_recovery_perc=performance_recovery_dp
                )
                st.success("Data sanitization mitigation complete and risk registered!")
