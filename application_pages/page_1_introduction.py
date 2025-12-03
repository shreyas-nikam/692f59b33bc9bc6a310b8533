import streamlit as st

def main():
    st.header("1. Introduction to AI Bill of Materials (AI-BOM) for Risk Management")
    st.markdown("""
    An AI Bill of Materials (AI-BOM) provides a structured inventory of all components (datasets, models, libraries, hardware) that comprise an AI system, along with their dependencies. This transparency is crucial for managing risks across the AI supply chain.

    For a complex AI system, understanding dependencies is paramount. Just as a software Bill of Materials (SBOM) tracks software components, an AI-BOM extends this concept to cover AI-specific elements like training data and model architectures. This helps in identifying potential vulnerabilities, assessing data provenance, and understanding the cascading impact of a compromised element.
    """)
    st.markdown("---")
    st.markdown("""
    This section lays the foundation for understanding what an AI-BOM is and why it's essential for AI risk management. Subsequent sections will build upon this by generating an AI-BOM, visualizing it, and analyzing risks.
    """)