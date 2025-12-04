import streamlit as st

# Import functions from utils
from utils import calculate_component_risk_aibom as calculate_component_risk, get_component_risk_profile, aggregate_overall_ai_system_risk


def main():
    st.header("5. Introduction to Vulnerability Scoring")
    st.markdown("""
    Vulnerability scoring provides a standardized way to quantify the severity of security weaknesses. The Common Vulnerability Scoring System (CVSS) is a widely used open framework for communicating these characteristics and impacts. A simplified version of CVSS helps in prioritizing risks.

    The CVSS score is derived from various metrics, broadly categorized into base, temporal, and environmental metrics. For our simplified scenario, we can represent this as:
    $$ \text{CVSS Score} = g(\text{AttackVector}, \text{AttackComplexity}, \text{PrivilegesRequired}, \dots) $$
    where $g$ is a function that combines several factors related to the vulnerability's exploitability and impact. In our synthetic data, 'Known_Vulnerabilities_Score' directly serves as this simplified score. A higher score indicates a more severe vulnerability.
    """)
    if not st.session_state.ai_bom_df.empty:
        max_vuln_score = st.session_state.ai_bom_df['Known_Vulnerabilities_Score'].max(
        )
        min_vuln_score = st.session_state.ai_bom_df['Known_Vulnerabilities_Score'].min(
        )
        avg_vuln_score = st.session_state.ai_bom_df['Known_Vulnerabilities_Score'].mean(
        )

        st.metric(label="Maximum Known Vulnerabilities Score",
                  value=f"{max_vuln_score:.2f}")
        st.metric(label="Minimum Known Vulnerabilities Score",
                  value=f"{min_vuln_score:.2f}")
        st.metric(label="Average Known Vulnerabilities Score",
                  value=f"{avg_vuln_score:.2f}")
        st.markdown("""
        Understanding the range and distribution of vulnerability scores in our AI-BOM gives us a baseline for assessing the security posture of individual components. This numerical value is a direct input for our component risk calculations.
        """)

        st.header("6. Calculating Individual Component Risk")
        st.markdown("""
        Each component in the AI system carries its own set of risks, influenced by attributes like its type, origin, and known vulnerabilities. To assess these individual risks, we combine these attributes into a single 'Component Risk' score. For simplicity, we'll define a risk function that considers the 'Known_Vulnerabilities_Score' and potentially the 'Origin' (e.g., third-party components might inherently carry higher risk).
        """)
        st.markdown(
            "First 5 components with their vulnerability and calculated risk scores:")
        st.dataframe(st.session_state.ai_bom_df[[
                     'Component_ID', 'Component_Name', 'Known_Vulnerabilities_Score', 'Component_Risk_Score']].head())
        st.markdown("""
        We have now quantified the individual risk associated with each component. This 'Component_Risk_Score' provides a granular view of where potential problems might lie within the AI system, taking into account both reported vulnerabilities and supply chain factors.
        """)

        st.header("7. Displaying Component Risk Profiles")
        st.markdown("""
        A textual 'Risk Profile' provides a concise summary of a component's key risk-relevant information. This allows Risk Managers to quickly understand the implications of a specific component's presence in the AI system without needing to parse raw data. The profile should highlight origin, vulnerabilities, and the calculated component risk score.
        """)

        selected_component_id_profile = st.selectbox(
            "Select a component to view its Risk Profile:",
            options=st.session_state.ai_bom_df['Component_ID'].tolist(),
            key='component_profile_selector'
        )
        if selected_component_id_profile:
            risk_profile = get_component_risk_profile(
                st.session_state.ai_bom_df, selected_component_id_profile)
            st.text(risk_profile)
        st.markdown("""
        The component risk profile offers a quick and comprehensive summary, allowing Risk Managers to efficiently assess specific components. This is crucial for focused investigations and for understanding the details behind a component's assigned risk score.
        """)

        st.header("8. Aggregating Overall AI System Risk")
        st.markdown("""
        The overall AI System Risk is not just the sum of individual component risks. It must also account for how vulnerabilities can propagate through dependencies. A critical vulnerability in a foundational library, for example, could impact multiple models or data processing steps that depend on it.

        We define the overall risk aggregation function as:
        $$ \text{Overall AI System Risk} = f(\text{ComponentRisk}_1, \dots, \text{ComponentRisk}_N, \text{Interdependencies}) $$
        where $f$ is an aggregation function. A simple aggregation might be the maximum risk score found in any component, or a weighted average that gives more weight to components with many downstream dependencies. For this lab, we will use a weighted sum, where components with higher 'out-degrees' (more downstream dependencies) contribute more significantly to the overall risk if their individual risk is high.
        """)
        overall_risk_score = aggregate_overall_ai_system_risk(
            st.session_state.ai_system_graph)
        st.metric(label="Overall Calculated AI System Risk Score (0-100)",
                  value=f"{overall_risk_score:.2f}")
        st.markdown("""
        By aggregating the individual component risks and considering their interdependencies, we arrive at a more holistic view of the AI system's security posture. This overall score helps in strategic decision-making regarding risk tolerance and resource allocation for mitigation efforts.
        """)
    else:
        st.info(
            "Generate an AI-BOM to see vulnerability scoring, component risks, and overall system risk.")
