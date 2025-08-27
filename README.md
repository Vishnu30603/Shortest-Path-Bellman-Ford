# Smart Route Optimizer (Shortest Path with Bellman-Ford)

An interactive **Streamlit web app** to visualize, edit, and optimize custom road networks using the **Bellman-Ford algorithm**.  
The app accounts for **traffic conditions** and **road quality**, making it more realistic for transportation and routing problems.

---

## Features

- **Upload CSV files** describing your custom road network  
- **Edit and preview data** directly inside the app  
- **Bellman-Ford Algorithm** for shortest path calculation with **negative weight handling**  
- **Negative cycle detection** with user alerts  
- **Condition adjustments**:
  - Traffic (`Low`, `Medium`, `High`) → alters travel cost  
  - Road quality (`Smooth`, `Rough`) → alters travel cost  
- **Top-K shortest paths** (up to 3) between selected nodes  
- **Edge-by-edge breakdown** of chosen paths with adjusted weights  
- **Interactive graph visualization** using **Plotly**  
- **Customizable graph styling** (node size, colors, fonts, etc.)  

---

## Input Format

Upload a CSV file with the following required columns:

| Column       | Description                                      | Example       |
|--------------|--------------------------------------------------|---------------|
| **Source**   | Starting vertex of the edge                      | A             |
| **Destination** | Ending vertex of the edge                     | B             |
| **Weight**   | Base distance/weight of the road (numeric)       | 12.5          |
| **Traffic**  | Traffic condition (`Low`, `Medium`, `High`)      | High          |
| **Road**     | Road quality (`Smooth`, `Rough`)                 | Rough         |

---
## Route Weight Factors

| Factor        | Condition      | Multiplier | Impact               |
|:------------- |:--------------|:----------:|:---------------------|
| **Traffic**   | Low            | × 0.8      | Faster, minimal delay   |
|               | Medium         | × 1.0      | Normal flow             |
|               | High           | × 1.4      | Slower due to congestion |
| **Road**      | Smooth         | × 0.9      | Slightly faster         |
|               | Rough          | × 1.2      | Slower, less comfortable |

---
