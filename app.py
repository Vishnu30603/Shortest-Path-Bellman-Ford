import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

def create_graph(data):
    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(row['Source'], row['Destination'], weight=row['Weight'], traffic=row['Traffic'], road=row['Road'])
    return G

def bellman_ford(graph, source):
    dist = {node: float('inf') for node in graph.nodes}
    dist[source] = 0
    predecessors = {node: None for node in graph.nodes}

    for _ in range(len(graph.nodes) - 1):
        for u, v, data in graph.edges(data=True):
            weight = data['weight']
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                predecessors[v] = u

    # Check for negative-weight cycles
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        if dist[u] != float('inf') and dist[u] + weight < dist[v]:
            st.error("Graph contains a negative-weight cycle")
            return None, None

    return dist, predecessors

def adjust_for_conditions(graph, node_labels):
    adjusted_graph = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        traffic = data['traffic']
        road = data['road']
        traffic_adjustment = {'Low': -0.2, 'Medium': 0, 'High': 0.4}[traffic]
        road_adjustment = {'Smooth': -0.1, 'Rough': 0.3}[road]

        new_weight = data['weight'] + traffic_adjustment + road_adjustment
        adjusted_graph.add_edge(u, v, weight=new_weight)
    
    return adjusted_graph

def get_shortest_path(predecessors, source, target):
    path = []
    while target is not None:
        path.insert(0, target)
        target = predecessors[target]
    return path if path[0] == source else []

st.title("Smart Route Optimizer")
st.write("Upload a CSV file with columns: Source, Destination, Weight, Traffic, Road")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = load_data(uploaded_file)
    
    st.write("Edit Data:")
    edited_data = st.data_editor(data)
    
    st.write("Edited Data Preview:")
    st.dataframe(edited_data)

    num_vertices = len(set(edited_data['Source']).union(set(edited_data['Destination'])))
    vertex_names = [f'V{i}' for i in range(num_vertices)]

    st.write("Optional: Provide custom names for vertices (comma-separated, e.g., P,Q,R,S,T,U):")
    custom_names_input = st.text_input("Custom Vertex Names")
    if custom_names_input:
        custom_names = custom_names_input.split(',')
        if len(custom_names) == num_vertices:
            vertex_names = custom_names
        else:
            st.warning(f"Number of custom names provided ({len(custom_names)}) does not match the number of vertices ({num_vertices}). Using default names.")

    G = create_graph(edited_data)
    source = 0  # You can modify this as needed

    # Map node indices to names
    node_labels = {i: vertex_names[i] for i in range(num_vertices)}

    # Customization options
    st.sidebar.header("Graph Customization Options")
    plot_width = st.sidebar.slider("Plot width", 5, 20, 10)
    plot_height = st.sidebar.slider("Plot height", 5, 20, 6)
    node_size = st.sidebar.slider("Node size", 100, 5000, 2000)
    font_size = st.sidebar.slider("Font size", 10, 50, 15)
    node_color = st.sidebar.color_picker("Node color", "#ADD8E6")  # Default to light blue
    edge_color = st.sidebar.color_picker("Edge color", "#999999")
    path_color = st.sidebar.color_picker("Path color", "#FF0000")
    
    # Visualize Initial Graph
    st.subheader("Initial Graph")
    pos = nx.spring_layout(G, weight='weight')  # Use spring layout
    plt.figure(figsize=(plot_width, plot_height))
    nx.draw(G, pos, labels=node_labels, with_labels=True, node_color=node_color, edge_color=edge_color, node_size=node_size, font_size=font_size)
    labels = nx.get_edge_attributes(G, 'weight')
    labels = {k: round(v, 2) for k, v in labels.items()}  # Round off edge weights to 2 decimal places
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=font_size)
    st.pyplot(plt)

    dist, predecessors = bellman_ford(G, source)
    if dist is not None:
        dist = {node_labels[k]: v for k, v in dist.items()}  # Map distances to custom names
        st.write("Initial shortest distances from source:", {k: round(v, 2) for k, v in dist.items()})

        # Adjust the graph for conditions
        adjusted_G = adjust_for_conditions(G, node_labels)
        adjusted_dist, adjusted_predecessors = bellman_ford(adjusted_G, source)
        if adjusted_dist is not None:
            adjusted_dist = {node_labels[k]: v for k, v in adjusted_dist.items()}  # Map distances to custom names
            adjusted_dist = {k: round(v, 2) for k, v in adjusted_dist.items()}  # Round off distances to 2 decimal places
            st.write("Adjusted shortest distances considering traffic and road conditions:", adjusted_dist)

            target = st.selectbox("Select the destination vertex:", options=list(node_labels.values()))
            target_idx = list(node_labels.keys())[list(node_labels.values()).index(target)]
            path = get_shortest_path(adjusted_predecessors, source, target_idx)
            path_names = [node_labels[node] for node in path]
            st.write(f"Shortest path from {node_labels[source]} to {target}:", " -> ".join(path_names))

            # Output the shortest distance for the selected path
            if path:
                total_distance = sum(adjusted_G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                st.write(f"Shortest distance for the path from {node_labels[source]} to {target} considering adjusted weights: {round(total_distance, 2)}")

            # Visualize Final Graph with Path
            st.subheader("Final Graph with Shortest Path")
            plt.figure(figsize=(plot_width, plot_height))
            nx.draw(adjusted_G, pos, labels=node_labels, with_labels=True, node_color=node_color, edge_color=edge_color, node_size=node_size, font_size=font_size)
            nx.draw_networkx_edges(adjusted_G, pos, edgelist=[(path[i], path[i+1]) for i in range(len(path)-1)], edge_color=path_color, width=2)
            labels = nx.get_edge_attributes(adjusted_G, 'weight')
            labels = {k: round(v, 2) for k, v in labels.items()}  # Round off edge weights to 2 decimal places
            nx.draw_networkx_edge_labels(adjusted_G, pos, edge_labels=labels, font_size=font_size)
            st.pyplot(plt)





























