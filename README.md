# Shortest-Path-Bellman-Ford
# Problem statement:
Suppose you are given a directed graph with 6 vertices P, Q, R, S, T, and U, where each vertex represents a location in a city. The graph has 8 edges with their respective weights and assigned values, and each edge represents a road connecting two locations.
The assigned values are as follows:                                                                                                     
P --> 0                                                                                                                               
Q --> 1                                                                                                                               
R --> 2                                                                                                                               
S --> 3                                                                                                                               
T --> 4                                                                                                                               
U --> 5                                                                                                                               
The edges and their respective assigned values are:                                                                                     
Edge 1: P-Q, weight=4                                                                                                                 
Edge 2: P-R, weight=8                                                                                                                 
Edge 3: P-S, weight=5                                                                                                                
Edge 4: Q-T, weight=1                                                                                                                
Edge 5: R-T, weight=9                                                                                                                
Edge 6: S-R, weight=2                                                                                                                
Edge 7: S-U, weight=4                                                                                                                
Edge 8: U-T, weight=3                                                                                                                
You need to find out the shortest distance of each vertex from the source vertex and also the quickest path vertex from the source using the Bellman-Ford algorithm. In addition, you need to take into account the traffic and road conditions from the source vertex to each vertex.                                                                                                                            
The traffic and road conditions from the source vertex to each vertex are given as follows:                                             
From P to Q: high-traffic (h), uneven road (r)                                                                                        
From P to R: low-traffic (l), even road (s)                                                                                           
From P to S: moderate traffic (m), even road (s)                                                                                      
From P to T: low-traffic (l), uneven road (r)                                                                                         
From P to U: high-traffic (h), even road (s)                                                                                          
