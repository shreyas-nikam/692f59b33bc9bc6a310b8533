import streamlit as st
import io


def main():
    st.header("2. Generating a Synthetic AI-BOM Dataset")
    st.markdown("""
    To demonstrate the utility of an AI-BOM, we will generate a synthetic dataset. This dataset will represent various components of an AI system and their interdependencies. Each component will have attributes crucial for risk assessment, such as type, origin, version, known vulnerabilities, and licensing information. The dependencies will define the flow and connections between these components.
    """)

    # Import necessary functions from utils
    from utils import generate_ai_bom_dataset, calculate_component_risk_aibom as calculate_component_risk, create_ai_system_graph

    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.num_components_input = st.slider(
            "Number of Components", min_value=5, max_value=50, value=st.session_state.num_components_input, key='num_components_slider')
    with col2:
        st.session_state.num_dependencies_input = st.slider(
            "Number of Dependencies", min_value=5, max_value=100, value=st.session_state.num_dependencies_input, key='num_dependencies_slider')

    if st.button("Generate AI-BOM", key='generate_button', type="primary"):
        # Call generate_ai_bom_dataset and update session state
        st.session_state.ai_bom_df, st.session_state.ai_bom_dependencies = generate_ai_bom_dataset(
            st.session_state.num_components_input,
            st.session_state.num_dependencies_input
        )
        # Re-calculate component risks and create initial graph
        st.session_state.ai_bom_df['Component_Risk_Score'] = st.session_state.ai_bom_df.apply(
            calculate_component_risk, axis=1)
        st.session_state.ai_system_graph = create_ai_system_graph(
            st.session_state.ai_bom_df, st.session_state.ai_bom_dependencies)
        st.session_state.simulated_risk_graph = st.session_state.ai_system_graph.copy()
        st.success("AI-BOM Generated!")
        st.rerun()

    st.divider()
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
        st.info(
            "Please generate an AI-BOM using the sidebar controls to see the dataset and its attributes.")
