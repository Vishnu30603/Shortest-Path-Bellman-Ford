#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// Struct to represent an edge in the graph
struct Edge
{
    int source, destination, weight;
};

// Struct to represent a graph
struct Graph
{
    int vertices, edges;
    struct Edge *edge;
};

// Function to create a graph with V vertices and E edges
struct Graph *createGraph(int vertices, int edges)
{
    struct Graph *graph = (struct Graph *)malloc(sizeof(struct Graph));
    graph->vertices = vertices;
    graph->edges = edges;
    graph->edge = (struct Edge *)malloc(edges * sizeof(struct Edge));
    return graph;
}

// Function to print the distance array returned by the Bellam-Ford algorithm
void printDistance(int dist[], int n)
{
    printf("Vertex\tShortest distance from the source\n");
    for (int i = 0; i < n; ++i)
    {
        if (i == 0)
            printf("Source\t\t0\n");
        else
            printf("%d\t\t%d\n", i, dist[i]);
    }
}
// Function to consider the dynamically varying parameters
// eg; Traffic congestion, condition of the road
void Traffic_And_Condition(int dist[], int n)
{
    int min_val = INT_MAX, min_vertex = -1;
    for (int i = 1; i < n; ++i)
    {
        char traffic, condition;
        printf("Enter the traffic(l/m/h) and road condition(s/r) from source to vertex %d (l/m/h):\n", i);
        scanf(" %c %c", &traffic, &condition);
        int t, c;
        if (traffic == 'l')
            t = -1;
        else if (traffic == 'm')
            t = 0;
        else if (traffic == 'h')
            t = 1;
        if (condition == 's')
            c = -1;
        else if (condition == 'r')
            c = 1;

        int new_dist = dist[i] + t + c;
        if (new_dist < min_val)
        {
            min_val = new_dist;
            min_vertex = i;
        }
    }
    printf("The path from the source to vertex %d is the quickest\n", min_vertex);
}
// Bellman-Ford algorithm to find the shortest path from a source vertex to all other vertices
void BellmanFord(struct Graph *graph, int source)
{
    int V = graph->vertices;
    int E = graph->edges;
    int dist[V];

    // Initialize distance array
    for (int i = 0; i < V; ++i)
    {
        dist[i] = INT_MAX;
    }
    dist[source] = 0;

    // Relax all edges V-1 times
    for (int i = 1; i <= V - 1; ++i)
    {
        for (int j = 0; j < E; ++j)
        {
            int u = graph->edge[j].source;
            int v = graph->edge[j].destination;
            int weight = graph->edge[j].weight;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
            {
                dist[v] = dist[u] + weight;
            }
        }
    }

    // Check for negative-weight cycles
    for (int i = 0; i < E; ++i)
    {
        int u = graph->edge[i].source;
        int v = graph->edge[i].destination;
        int weight = graph->edge[i].weight;
        if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
        {
            printf("The graph contains a negative-weight cycle\n");
            return;
        }
    }

    // Print distance array
    printDistance(dist, V);
    Traffic_And_Condition(dist, V);
}

int main()
{
    int V, E, s = 0;
    printf("Enter the number of vertices and edges in the graph :\n");
    scanf("%d%d", &V, &E);

    struct Graph *graph = createGraph(V, E);
    for (int i = 0; i < E; ++i)
    {
        int u, v, w;
        printf("Enter the source, destination and weight of edge %d :\n", i + 1);
        scanf("%d%d%d", &u, &v, &w);
        graph->edge[i].source = u;
        graph->edge[i].destination = v;
        graph->edge[i].weight = w;
    }

    BellmanFord(graph, s);

    return 0;
}
