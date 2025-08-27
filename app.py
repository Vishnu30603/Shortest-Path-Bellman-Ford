import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# Data Validation
REQUIRED_COLUMNS = {'Source', 'Destination', 'Weight', 'Traffic', 'Road'}

def validate_data(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    if df.isnull().any().any():
        return False, "Data contains missing values."
    if not all(df['Weight'].apply(lambda x: isinstance(x, (int, float)))):
        return False, "Weights must be numeric."
    if not all(df['Traffic'].isin(['Low', 'Medium', 'High'])):
        return False, "Traffic must be one of: Low, Medium, High."
    if not all(df['Road'].isin(['Smooth', 'Rough'])):
        return False, "Road must be one of: Smooth, Rough."
    return True, ""

def create_graph(data):
    G = nx.DiGraph()
    for _, row in data.iterrows():
        G.add_edge(
            row['Source'], row['Destination'],
            weight=row['Weight'],
            traffic=row['Traffic'],
            road=row['Road']
        )
    return G

def adjust_for_conditions(graph):
    # Proportional Adjustments
    traffic_factor = {'Low': 0.8, 'Medium': 1.0, 'High': 1.4}
    road_factor = {'Smooth': 0.9, 'Rough': 1.2}
    adjusted_graph = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        tf = traffic_factor.get(data['traffic'], 1.0)
        rf = road_factor.get(data['road'], 1.0)
        new_weight = data['weight'] * tf * rf
        adjusted_graph.add_edge(
            u, v,
            weight=new_weight,
            base_weight=data['weight'],
            traffic=data['traffic'],
            road=data['road'],
            traffic_factor=tf,
            road_factor=rf
        )
    return adjusted_graph

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
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        if dist[u] != float('inf') and dist[u] + weight < dist[v]:
            return None, None, "Graph contains a negative-weight cycle."
    return dist, predecessors, ""

def k_shortest_paths(graph, source, target, k=3):
    try:
        paths = list(nx.shortest_simple_paths(graph, source, target, weight='weight'))
        return paths[:k]
    except nx.NetworkXNoPath:
        return []

def ordinal(n):
    # Returns 1st, 2nd, 3rd, etc.
    return "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])

def display_path_breakdown(path, adjusted_G, node_labels):
    rows = []
    for u, v in zip(path[:-1], path[1:]):
        edge = adjusted_G[u][v]
        rows.append({
            "From": node_labels[u],
            "To": node_labels[v],
            "Base": edge['base_weight'],
            "Traffic ×": f"{edge['traffic_factor']} ({edge['traffic']})",
            "Road ×": f"{edge['road_factor']} ({edge['road']})",
            "Final Weight": round(edge['weight'], 2)
        })
    df = pd.DataFrame(rows)
    st.markdown("**Edge-by-Edge Breakdown:**")
    st.table(df)

def plot_graph(graph, paths=None, node_labels=None, node_color="#ADD8E6", edge_color="#888", path_colors=None, node_size=20, font_size=14, plot_width=700, plot_height=500):
    pos = nx.spring_layout(graph, weight='weight', seed=42)
    edge_x, edge_y = [], []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, 
        line=dict(width=1, color=edge_color),
        hoverinfo='none', mode='lines'
    )

    node_x, node_y, text = [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label = node_labels[node] if node_labels else str(node)
        text.append(label)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=text, textposition="top center",
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color='DarkSlateGrey')),
        hoverinfo='text',
        textfont=dict(size=font_size)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        width=plot_width,
                        height=plot_height
                    ))
    if paths:
        if path_colors is None:
            path_colors = ['#FF0000', '#00AA00', '#0000FF']
        for idx, path in enumerate(paths):
            px, py = [], []
            for u, v in zip(path[:-1], path[1:]):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                px += [x0, x1, None]
                py += [y0, y1, None]
            fig.add_trace(go.Scatter(
                x=px, y=py, mode='lines',
                line=dict(width=4, color=path_colors[idx % len(path_colors)]),
                name=f'Path {idx+1}'
            ))
    return fig

# Streamlit App
st.set_page_config(
    page_title="Smart Route Optimizer",
    page_icon="✅",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center; font-weight: 700; margin-bottom: 0.2em;'>Smart Route Optimizer</h1>"
    "<h4 style='text-align: center; color: #888; font-weight: 400;'>"
    "Visualize, edit, and optimize your custom road networks with ease"
    "</h4><br>",
    unsafe_allow_html=True
)

st.write("Upload a CSV file with columns: Source, Destination, Weight, Traffic, Road")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    valid, msg = validate_data(data)
    if not valid:
        st.error(f"Data Error: {msg}")
        st.stop()

    st.write("Edit Data:")
    edited_data = st.data_editor(data)
    st.write("Preview Data:")
    st.dataframe(edited_data)

    vertices = sorted(set(edited_data['Source']).union(set(edited_data['Destination'])))
    num_vertices = len(vertices)
    vertex_names = [f'V{i}' for i in range(num_vertices)]
    st.write("Optional: Provide custom names for vertices (comma-separated, e.g., P,Q,R,S,T,U):")
    custom_names_input = st.text_input("Custom Vertex Names")
    if custom_names_input:
        custom_names = [x.strip() for x in custom_names_input.split(',')]
        if len(custom_names) == num_vertices:
            vertex_names = custom_names
        else:
            st.warning(f"Number of custom names provided ({len(custom_names)}) does not match the number of vertices ({num_vertices}). Using default names.")
    node_labels = dict(zip(vertices, vertex_names))

    # Sidebar Customization Controls
    st.sidebar.header("Graph Customization Options")
    plot_width = st.sidebar.slider("Plot width", 500, 1200, 700)
    plot_height = st.sidebar.slider("Plot height", 300, 900, 500)
    node_size = st.sidebar.slider("Node size", 10, 50, 20)
    font_size = st.sidebar.slider("Font size", 10, 30, 14)
    node_color = st.sidebar.color_picker("Node color", "#ADD8E6")
    edge_color = st.sidebar.color_picker("Edge color", "#888888")
    path_color1 = st.sidebar.color_picker("Path 1 color", "#FF0000")
    path_color2 = st.sidebar.color_picker("Path 2 color", "#00AA00")
    path_color3 = st.sidebar.color_picker("Path 3 color", "#0000FF")
    path_colors = [path_color1, path_color2, path_color3]

    # Source/Destination selection
    st.subheader("Select Source and Destination")
    source_name = st.selectbox("Source Vertex", options=vertex_names)
    target_name = st.selectbox("Destination Vertex", options=vertex_names)
    source = vertices[vertex_names.index(source_name)]
    target = vertices[vertex_names.index(target_name)]

    # Create and adjust graph
    G = create_graph(edited_data)
    adjusted_G = adjust_for_conditions(G)

    # Bellman-Ford and negative cycle check
    dist, predecessors, err = bellman_ford(adjusted_G, source)
    if err:
        st.error(err)
        st.stop()

    # Compute top-k shortest paths and let user select which to display
    k = 3
    paths = k_shortest_paths(adjusted_G, source, target, k=k)
    if not paths:
        st.warning("No path found between selected nodes.")
    else:
        path_options = [f"{ordinal(i+1)} shortest path" for i in range(len(paths))]
        selected_idx = st.selectbox("Which shortest path to display?", options=range(len(paths)), format_func=lambda i: path_options[i])
        selected_path = paths[selected_idx]
        path_names = [node_labels[n] for n in selected_path]
        total = sum(adjusted_G[u][v]['weight'] for u, v in zip(selected_path[:-1], selected_path[1:]))
        st.subheader(f"{path_options[selected_idx]}: {' → '.join(path_names)} (Total Distance: {round(total, 2)})")

        display_path_breakdown(selected_path, adjusted_G, node_labels)

        # Interactive graph for selected path
        st.subheader("Interactive Graph Visualization")
        fig = plot_graph(
            adjusted_G, [selected_path], node_labels,
            node_color=node_color,
            edge_color=edge_color,
            path_colors=[path_colors[selected_idx]],
            node_size=node_size,
            font_size=font_size,
            plot_width=plot_width,
            plot_height=plot_height
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
### Route Weight Factors

| Factor        | Condition      | Multiplier | Impact               |
|:------------- |:--------------|:----------:|:---------------------|
| **Traffic**   | Low            | × 0.8      | Faster, minimal delay   |
|               | Medium         | × 1.0      | Normal flow             |
|               | High           | × 1.4      | Slower due to congestion |
| **Road**      | Smooth         | × 0.9      | Slightly faster         |
|               | Rough          | × 1.2      | Slower, less comfortable |
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.info("""
**Why Bellman-Ford Algorithm is Used?**

- Handles negative edge weights
- Detects negative cycles in the graph
- Stops and alerts the user if a negative cycle is found
""")
