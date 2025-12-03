import streamlit as st
import pandas as pd


def main():
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
        data_components_df = st.session_state.ai_bom_df[st.session_state.ai_bom_df['Component_Type'] == 'Data'].copy(
        )
        st.subheader("Data Components and their Risk-Relevant Attributes:")
        st.dataframe(data_components_df[[
                     'Component_ID', 'Origin', 'Licensing_Info', 'Component_Risk_Score']])
        if not data_components_df.empty:
            mean_data_risk = data_components_df['Component_Risk_Score'].mean()
            st.write(
                f"Average Component Risk Score for 'Data' components: **{mean_data_risk:.2f}**")
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
        third_party_origins = ['Third-Party Vendor A',
                               'Third-Party Vendor B', 'Open Source Community']
        third_party_components_df = st.session_state.ai_bom_df[st.session_state.ai_bom_df['Origin'].isin(
            third_party_origins)].copy()
        st.subheader(
            "Third-Party Components and their Risk-Relevant Attributes:")
        st.dataframe(third_party_components_df[[
                     'Component_ID', 'Component_Type', 'Origin', 'Known_Vulnerabilities_Score', 'Component_Risk_Score']])
        if not third_party_components_df.empty:
            mean_third_party_risk = third_party_components_df['Component_Risk_Score'].mean(
            )
            st.write(
                f"Average Component Risk Score for third-party components: **{mean_third_party_risk:.2f}**")
        else:
            st.info("No 'Third-Party' components found in the current AI-BOM.")
        st.markdown("""
        This targeted analysis of third-party components highlights their contribution to the overall risk profile. Understanding these external risks is vital for implementing robust vetting processes, contractual agreements, and continuous monitoring for components originating outside the organization's direct control.
        """)

        st.subheader("Top and Bottom N Components by Risk Score")
        st.markdown("""
        Identify the highest and lowest risk components to understand where to focus vulnerability management efforts.
        """)
        st.session_state.num_top_bottom_n_input = st.number_input(
            "Enter N for Top/Bottom Components:",
            min_value=1, max_value=len(st.session_state.ai_bom_df), value=st.session_state.num_top_bottom_n_input, key='top_bottom_n_input'
        )

        if not st.session_state.ai_bom_df.empty:
            st.markdown(
                f"Top {st.session_state.num_top_bottom_n_input} components with the highest risk scores:")
            st.dataframe(st.session_state.ai_bom_df.sort_values(by='Component_Risk_Score', ascending=False).head(
                st.session_state.num_top_bottom_n_input)[['Component_ID', 'Component_Name', 'Component_Type', 'Origin', 'Component_Risk_Score']])

            st.markdown(
                f"\nBottom {st.session_state.num_top_bottom_n_input} components with the lowest risk scores:")
            st.dataframe(st.session_state.ai_bom_df.sort_values(by='Component_Risk_Score', ascending=True).head(
                st.session_state.num_top_bottom_n_input)[['Component_ID', 'Component_Name', 'Component_Type', 'Origin', 'Component_Risk_Score']])
        st.markdown("""
        By identifying the highest and lowest risk components, we can understand where to focus our vulnerability management efforts. High-risk components, especially those from third-parties or with many dependencies, warrant immediate attention for mitigation, patching, or replacement, thereby improving the overall transparency and security of the AI supply chain.
        """)
    else:
        st.info("Generate an AI-BOM to perform data provenance, third-party, and top/bottom component risk analysis.")


st.markdown("---")
st.markdown("""
The insights gained from an AI-BOM are indispensable for maintaining the security, integrity, and trustworthiness of AI deployments in an increasingly complex threat landscape. Proactive risk identification and management are key to building resilient AI systems.
""")
