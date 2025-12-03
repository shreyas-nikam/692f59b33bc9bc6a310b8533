import streamlit as st
import io

def main():
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