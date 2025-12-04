import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import (
    generate_synthetic_image_data, define_simple_cnn_model, train_model,
    evaluate_model_performance, generate_adversarial_example,
    apply_mitigation_adversarial_training, calculate_risk_score,
    display_risk_status, create_performance_chart, add_to_risk_register,
    num_samples, img_size, epochs, lr, class_names
)


@st.cache_data(ttl="2h")
def get_base_data_adv():
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
def get_and_train_baseline_model_adv(train_loader):
    model = define_simple_cnn_model()
    train_model(model, train_loader, epochs, lr)
    return model


def main():
    st.header("Understanding AI Attack Vector 2: Adversarial Examples")
    st.markdown(r"""
    Adversarial examples are inputs to AI models that are intentionally designed by an attacker
    to cause the model to make a mistake. These perturbations are often imperceptible to humans
    but can drastically alter a model's prediction.

    In this simulation, we'll create an adversarial example for an image classifier and observe
    how a small, imperceptible change can lead to misclassification.

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

    X_train, y_train, X_test, y_test, train_loader, test_loader = get_base_data_adv()
    baseline_model_adv = get_and_train_baseline_model_adv(train_loader)
    baseline_accuracy_adv = evaluate_model_performance(
        baseline_model_adv, test_loader)

    st.markdown(f"**Baseline Model Accuracy:** `{baseline_accuracy_adv:.2f}%`")
    st.markdown(r"""
    *Annotation:* This metric represents the percentage of correctly classified images by the model.
    """)

    st.subheader("Adversarial Example Attack Scenario")
    st.markdown(r"""
    We will generate an adversarial example for a randomly selected image from the test set.
    Observe how a small perturbation, controlled by epsilon, can change the model's prediction.
    """)

    epsilon = st.slider("Select Epsilon (Attack Strength)",
                        0.0, 0.5, 0.1, 0.01)
    initial_p_event_ae = st.selectbox(
        "Initial P(Event) - Adversarial Examples",
        options=["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index("Medium"),
        key="ae_p_event_initial"
    )
    initial_m_consequence_ae = st.selectbox(
        "Initial M(Consequence) - Adversarial Examples",
        options=["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index("High"),
        key="ae_m_consequence_initial"
    )

    if st.button("Generate Adversarial Example", key="run_ae_scenario"):
        with st.spinner("Generating adversarial example..."):
            # Pick a random image from test set
            idx = np.random.randint(0, len(X_test))
            original_image = X_test[idx]
            original_label = y_test[idx]

            # Generate adversarial image
            adversarial_image_tensor = generate_adversarial_example(
                baseline_model_adv, torch.tensor(
                    original_image, dtype=torch.float32),
                original_label, epsilon=epsilon
            )
            adversarial_image_np = adversarial_image_tensor.cpu().numpy()

            # Predict with baseline model
            baseline_model_adv.eval()
            with torch.no_grad():
                original_output = baseline_model_adv(torch.tensor(
                    original_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
                _, original_prediction = torch.max(original_output.data, 1)
                adv_output = baseline_model_adv(
                    adversarial_image_tensor.unsqueeze(0).unsqueeze(0))
                _, adv_prediction = torch.max(adv_output.data, 1)

            st.subheader("Original vs. Adversarial Image")
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    original_image, caption=f"Original (True: {class_names[original_label]}, Pred: {class_names[original_prediction.item()]})", width=200)
            with col2:
                st.image(adversarial_image_np,
                         caption=f"Adversarial (True: {class_names[original_label]}, Pred: {class_names[adv_prediction.item()]})", width=200)

            # For overall performance impact due to adversarial examples
            # Create an adversarial test set for a more robust evaluation
            adv_X_test_subset = []
            adv_y_test_subset = []
            # Generate a small subset of adversarial examples for demonstration
            # Ensure we don't exceed test set size
            num_adv_samples_eval = min(50, len(X_test) // 2)
            if num_adv_samples_eval > 0:
                for i in np.random.choice(len(X_test), num_adv_samples_eval, replace=False):
                    current_image = X_test[i]
                    current_label = y_test[i]
                    adv_img = generate_adversarial_example(baseline_model_adv, torch.tensor(
                        current_image, dtype=torch.float32), current_label, epsilon=epsilon)
                    adv_X_test_subset.append(adv_img.cpu().numpy())
                    adv_y_test_subset.append(current_label)

                adv_X_test_subset_tensor = torch.tensor(
                    np.array(adv_X_test_subset), dtype=torch.float32)
                adv_y_test_subset_tensor = torch.tensor(
                    np.array(adv_y_test_subset), dtype=torch.long)
                adv_test_dataset_subset = TensorDataset(
                    adv_X_test_subset_tensor, adv_y_test_subset_tensor)
                adv_test_loader_subset = DataLoader(
                    adv_test_dataset_subset, batch_size=32, shuffle=False)

                adversarial_accuracy = evaluate_model_performance(
                    baseline_model_adv, adv_test_loader_subset)
            else:
                adversarial_accuracy = baseline_accuracy_adv  # No adversarial samples generated

            st.session_state["ae_adversarial_accuracy"] = adversarial_accuracy
            # Use these for training
            st.session_state["ae_adv_X_train_for_mitigation"] = adv_X_test_subset
            st.session_state["ae_adv_y_train_for_mitigation"] = adv_y_test_subset
            # Store original loader
            st.session_state["ae_original_train_loader"] = train_loader
            # Store test loader
            st.session_state["ae_test_loader"] = test_loader

            st.subheader("Results After Adversarial Attack")
            st.markdown(
                f"**Accuracy on Adversarial Examples:** `{adversarial_accuracy:.2f}%`")
            performance_impact_ae = baseline_accuracy_adv - adversarial_accuracy
            st.markdown(r"""
            *Annotation:* **Performance Impact (%)** measures the percentage decrease in model accuracy
            due to the attack: $$ \text{Impact} = \text{Baseline Accuracy} - \text{Attacked Accuracy} $$.
            """)
            st.markdown(
                f"**Performance Impact:** `{performance_impact_ae:.2f}%`")

            post_attack_p_event_ae = "High" if performance_impact_ae > 20 else "Medium"
            post_attack_m_consequence_ae = "High" if performance_impact_ae > 20 else "Medium"
            post_attack_risk_score_ae = calculate_risk_score(
                post_attack_p_event_ae, post_attack_m_consequence_ae)

            st.markdown(display_risk_status(
                post_attack_risk_score_ae), unsafe_allow_html=True)

            metrics_ae = {
                "Baseline Accuracy": baseline_accuracy_adv,
                "Adversarial Accuracy": adversarial_accuracy
            }
            st.pyplot(create_performance_chart(
                metrics_ae, "Model Accuracy: Baseline vs. Adversarial"))

            st.session_state["ae_post_attack_p_event"] = post_attack_p_event_ae
            st.session_state["ae_post_attack_m_consequence"] = post_attack_m_consequence_ae
            st.session_state["ae_performance_impact"] = performance_impact_ae
            # Store for mitigation
            st.session_state["ae_baseline_model_for_mitigation"] = baseline_model_adv
            st.session_state["ae_baseline_accuracy_for_mitigation"] = baseline_accuracy_adv

            st.success(
                "Adversarial example generation and evaluation complete!")

    if "ae_adversarial_accuracy" in st.session_state:
        st.subheader("Mitigation Strategy 2: Adversarial Training")
        st.markdown(r"""
        Adversarial training involves augmenting the training data with adversarial examples,
        making the model more robust to such attacks. The model learns to correctly classify
        both normal and perturbed inputs.
        """)

        mitigated_p_event_ae = st.selectbox(
            "Mitigated P(Event) - Adversarial Examples",
            options=["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index("Low"),
            key="ae_p_event_mitigated"
        )
        mitigated_m_consequence_ae = st.selectbox(
            "Mitigated M(Consequence) - Adversarial Examples",
            options=["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index("Low"),
            key="ae_m_consequence_mitigated"
        )

        if st.button("Apply Adversarial Training", key="apply_ae_mitigation"):
            with st.spinner("Applying adversarial training..."):
                # Re-train model with adversarial examples
                mitigated_model_ae = define_simple_cnn_model()

                apply_mitigation_adversarial_training(
                    mitigated_model_ae,
                    st.session_state["ae_original_train_loader"],
                    st.session_state["ae_adv_X_train_for_mitigation"],
                    st.session_state["ae_adv_y_train_for_mitigation"],
                    epochs,
                    lr
                )

                # Evaluate mitigated model on original test set and adversarial test set
                mitigated_accuracy_on_original_test = evaluate_model_performance(
                    mitigated_model_ae, st.session_state["ae_test_loader"])

                # Re-generate adversarial examples for the mitigated model for a fair comparison
                adv_X_test_mitigated = []
                adv_y_test_mitigated = []
                # Use the same number of samples as before
                num_adv_samples_eval = min(50, len(X_test) // 2)
                if num_adv_samples_eval > 0:
                    for i in np.random.choice(len(X_test), num_adv_samples_eval, replace=False):
                        current_image = X_test[i]
                        current_label = y_test[i]
                        adv_img = generate_adversarial_example(mitigated_model_ae, torch.tensor(
                            current_image, dtype=torch.float32), current_label, epsilon=epsilon)
                        adv_X_test_mitigated.append(adv_img.cpu().numpy())
                        adv_y_test_mitigated.append(current_label)

                    adv_X_test_mitigated_tensor = torch.tensor(
                        np.array(adv_X_test_mitigated), dtype=torch.float32)
                    adv_y_test_mitigated_tensor = torch.tensor(
                        np.array(adv_y_test_mitigated), dtype=torch.long)
                    adv_test_dataset_mitigated = TensorDataset(
                        adv_X_test_mitigated_tensor, adv_y_test_mitigated_tensor)
                    adv_test_loader_mitigated = DataLoader(
                        adv_test_dataset_mitigated, batch_size=32, shuffle=False)
                    mitigated_accuracy_on_adversarial = evaluate_model_performance(
                        mitigated_model_ae, adv_test_loader_mitigated)
                else:
                    mitigated_accuracy_on_adversarial = mitigated_accuracy_on_original_test

                st.session_state["ae_mitigated_accuracy"] = mitigated_accuracy_on_adversarial

                st.subheader("Results After Adversarial Training")
                st.markdown(
                    f"**Mitigated Model Accuracy (on adversarial examples):** `{mitigated_accuracy_on_adversarial:.2f}%`")

                performance_recovery_ae = mitigated_accuracy_on_adversarial - \
                    st.session_state["ae_adversarial_accuracy"]
                st.markdown(r"""
                *Annotation:* **Performance Recovery (%)** indicates the improvement in model accuracy
                after applying mitigation: $$ \text{Recovery} = \text{Mitigated Accuracy} - \text{Attacked Accuracy} $$.
                """)
                st.markdown(
                    f"**Performance Recovery:** `{performance_recovery_ae:.2f}%`")

                mitigated_risk_score_ae = calculate_risk_score(
                    mitigated_p_event_ae, mitigated_m_consequence_ae)
                st.markdown(display_risk_status(
                    mitigated_risk_score_ae), unsafe_allow_html=True)

                metrics_mitigated_ae = {
                    "Baseline Accuracy": st.session_state["ae_baseline_accuracy_for_mitigation"],
                    "Adversarial Accuracy": st.session_state["ae_adversarial_accuracy"],
                    "Mitigated Accuracy": mitigated_accuracy_on_adversarial
                }
                st.pyplot(create_performance_chart(metrics_mitigated_ae,
                          "Model Accuracy: Baseline, Adversarial, Mitigated"))

                add_to_risk_register(
                    attack_type="Adversarial Examples",
                    description=f"Generated adversarial examples with epsilon={epsilon:.2f} causing misclassification.",
                    initial_p_qual=initial_p_event_ae,
                    initial_m_qual=initial_m_consequence_ae,
                    post_attack_p_qual=st.session_state["ae_post_attack_p_event"],
                    post_attack_m_qual=st.session_state["ae_post_attack_m_consequence"],
                    mitigation_applied="Adversarial Training",
                    mitigated_p_qual=mitigated_p_event_ae,
                    mitigated_m_qual=mitigated_m_consequence_ae,
                    perf_impact_perc=st.session_state["ae_performance_impact"],
                    perf_recovery_perc=performance_recovery_ae
                )
                st.success(
                    "Adversarial training mitigation complete and risk registered!")
