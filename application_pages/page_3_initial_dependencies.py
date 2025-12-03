import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx


def main():
    st.header("4. Visualizing AI System Dependencies (Initial Graph)")
    st.markdown("""
    A graphical representation of the AI system, where components are nodes and dependencies are edges, significantly enhances our understanding of its structure. This visualization allows Risk Managers to quickly grasp the complexity and interconnections within the AI supply chain.
    """)
    if st.session_state.ai_system_graph.number_of_nodes() > 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(st.session_state.ai_system_graph, seed=42)

        component_types = [nx.get_node_attributes(st.session_state.ai_system_graph, 'Component_Type')[
            node] for node in st.session_state.ai_system_graph.nodes()]
        unique_types = list(set(component_types))
        colors = plt.cm.get_cmap('tab10', len(unique_types))
        color_map = {ctype: colors(i) for i, ctype in enumerate(unique_types)}
        node_colors = [color_map[nx.get_node_attributes(st.session_state.ai_system_graph, 'Component_Type')[
            node]] for node in st.session_state.ai_system_graph.nodes()]

        nx.draw_networkx_nodes(st.session_state.ai_system_graph, pos,
                               node_color=node_colors, node_size=3000, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(st.session_state.ai_system_graph, pos, edgelist=st.session_state.ai_system_graph.edges(
        ), edge_color='gray', arrowsize=20, ax=ax)
        nx.draw_networkx_labels(
            st.session_state.ai_system_graph, pos, font_size=8, font_weight='bold', ax=ax)

        prompt_handles = [plt.Line2D([0], [0], marker='o', color='w', label=ctype,
                                     markerfacecolor=color_map[ctype], markersize=10)
                          for ctype in unique_types]
        ax.legend(handles=prompt_handles, title="Component Type")

        ax.set_title("Initial AI System Dependency Graph (AI-BOM)")
        ax.axis('off')
        st.pyplot(fig)
        st.markdown("""
        The visual graph provides an immediate understanding of the AI system's architecture. We can see how different components are interconnected, which is fundamental for identifying critical paths and potential single points of failure. The color-coding by component type gives a quick overview of the system's composition.
        """)
    else:
        st.info("Generate an AI-BOM to visualize the initial dependency graph.")
