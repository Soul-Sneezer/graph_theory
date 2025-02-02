#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <queue>
#include <functional>
#include <algorithm>
#include <stack>
#include <climits>
#include <string>
#include <unordered_map>

const int TREE = 1;
const int BACK = 2;
const int FORWARD = 3;
const int CROSS = 4;

const int WHITE = 0;
const int BLACK = 1;
const int UNCOLORED = -1;

struct edgenode 
{
    int y;
    int weight;
    int flow;
    int residual;
    edgenode() : y(0), weight(0) {}
    edgenode(int y, int weight) : y(y), weight(weight), flow(0), residual(weight) {} 
};

struct edgepair 
{
    int x;
    int y;
    int weight;

    edgepair(int x, int y, int weight) : x(x), y(y), weight(weight) {}
};

struct adjacency_matrix 
{
    std::vector<std::vector<int>> weight;
    int nvertices;

    adjacency_matrix(int nvertices)
    {
        this->nvertices = nvertices;
        weight.resize(nvertices);
        for (int i = 0; i < nvertices; i++)
        {
           weight[i].resize(nvertices); 
        }

        for(int i = 0; i < nvertices; i++)
        {
            for(int j = 0; j < nvertices; j++)
            {
                weight[i][j] = INT_MAX;
            }
        }
    }
};

class UnionFind 
{
    private:
        std::vector<int> p;
        std::vector<int> size_;
    public:
        UnionFind(int n)
        {
            p.resize(n);
            size_.resize(n);
            for (int i = 0; i < n; i++)
            {
               p[i] = i;
               size_[i] = 1;
            }
        }

        int find(int x)
        {
            if (this->p[x] == x)
            {
                return x; 
            }

            return find(this->p[x]);
        }

        void union_sets(int s1, int s2)
        {
            int r1, r2;

            r1 = find(s1);
            r2 = find(s2);

            if (r1 == r2)
                return; 

            if (size_[r1] >= size_[r2])
            {
                this->size_[r1] = size_[r1] + size_[r2];
                this->p[r2] = r1;
            }
            else 
            {
                this->size_[r2] = size_[r1] + size_[r2];
                this->p[r1] = r2;
            }
        }

        bool same_component(int s1, int s2)
        {
            return (find(s1) == find(s2));
        }
};

class Graph 
{
private:
    bool weighted;
    bool directed;
    bool bipartite; 

    int nvertices;
        
    int bfs(std::vector<std::vector<int>>& rGraph, int s, int t, std::vector<int>& parent) {
        fill(parent.begin(), parent.end(), -1);
        std::queue<std::pair<int, int>> q;
        q.push({s, INT_MAX});
        while(!q.empty()) {
            auto [u, flow] = q.front(); q.pop();
            for(int v=0; v<rGraph.size(); ++v)
                if(parent[v] == -1 && rGraph[u][v] > 0) {
                    parent[v] = u;
                    int new_flow = std::min(flow, rGraph[u][v]);
                    if(v == t) return new_flow;
                    q.push({v, new_flow});
                }
        }
        return 0;
    }
public:
    std::vector<std::vector<edgenode>> edges;

    Graph()
    {
    }

    Graph(int nvertices, bool directed = false, bool weighted = false)
    {
        this->nvertices = nvertices;
        this->weighted = weighted;
        this->directed = directed;
        this->edges.resize(this->nvertices);
    }
    
    Graph(const Graph& other)
    {
        this->nvertices = other.nvertices;
        this->edges = other.edges;
        this->weighted = other.weighted;
        this->directed = other.directed;
    }

    void newVertex()
    {
        nvertices++;
    }
    
    void addEdge(int v, edgenode e, int weight = 0)
    {
        edges[v].emplace_back(e);
        edges[e.y].emplace_back(edgenode(v, weight));
    }

    void addEdge(int vertex1, int vertex2, int weight = 0)
    {
        if(vertex1 >= nvertices)
           throw std::out_of_range("Invalid vertex1 index"); 

        if(vertex2 >= nvertices)
           throw std::out_of_range("Invalid vertex2 index"); 

        edges[vertex1].emplace_back(edgenode(vertex2, weight));
        
        if (!directed)
            edges[vertex2].emplace_back(edgenode(vertex1, weight));
    }

    void BFS(int s,
            std::vector<bool>& discovered,
            std::vector<bool>& processed,
            std::vector<int>& parent,
            const std::function<void(int)>& process_vertex_early, 
            const std::function<void(int, int)>& process_edge, 
            const std::function<void(int)>& process_vertex_late) 
    {
        std::priority_queue<int> pq;
        pq.push(s);
        discovered[s] = true;
        
        while(!pq.empty())
        {
            int vertex = pq.top(); 
            std::cout<<vertex<<"\n";
            pq.pop();
            process_vertex_early(vertex);
            processed[vertex] = false; 

            for (edgenode edge : this->edges[vertex])
            {
                if ((!processed[edge.y]) || this->directed)
                {
                    process_edge(vertex, edge.y);
                }

                if (!discovered[edge.y])
                {
                    pq.push(edge.y);
                    discovered[edge.y] = true;
                    parent[edge.y] = vertex;
                }
            }

            process_vertex_late(vertex);

            process_vertex_late(vertex);
        }
    }

    int edge_classification(std::vector<int>& parent, 
                            std::vector<bool>& discovered, 
                            std::vector<bool>& processed, 
                            std::vector<int>& entry_time, 
                            int x, int y)
    {
        if (parent[y] == x)
            return (TREE);

        if (discovered[y] && !processed[y])
            return BACK;

        if (processed[y] && entry_time[y] > entry_time[x])
            return FORWARD;

        if (processed[y] && (entry_time[y] < entry_time[x]))
            return CROSS;

        printf("Warning self loop (%d %d)\n", x, y);

        return -1;
    }

    void find_path(int start, int end, std::vector<int>& parents)
    {
        if ((start == end) || (end == -1))
            printf("\n%d", start);
        else 
        {
            find_path(start, parents[end], parents);
            printf(" %d", end);
        }
    }

    void DFS(int s,
            int& __time,
            bool& finished,
            std::vector<bool>& discovered,
            std::vector<int>& entry_time,
            std::vector<bool>& processed,
            std::vector<int>& parent,
            std::vector<int>& exit_time,
            const std::function<void(int)>& process_vertex_early, 
            const std::function<void(int,int)>& process_edge, 
            const std::function<void(int)>& process_vertex_late)
    {
        if (finished)
            return;

        discovered[s] = true;
        __time = __time + 1;
        entry_time[s] = __time;

        process_vertex_early(s);
        
        for(edgenode edge : edges[s])
        {
            if (!discovered[edge.y])
            {
                DFS(edge.y, __time, finished, discovered, entry_time, processed, parent, exit_time, process_vertex_early, process_edge, process_vertex_late);
            }
            else if (((!processed[edge.y]) && (parent[s] != edge.y)) || (this->directed))
                process_edge(s, edge.y);

            if (finished) 
                return;
        }
        process_vertex_late(s);
        __time = __time + 1;
        exit_time[s] = __time;
        processed[s] = true;
    }

    void detectCyles()
    {
        int __time = 0;
        bool finished = false;
        std::vector<int> reachable_ancestor(this->nvertices);
        std::vector<int> parent(this->nvertices);
        std::vector<int> tree_out_degree(this->nvertices);
        std::vector<int> entry_time(this->nvertices);
        std::vector<bool> discovered(this->nvertices);
        std::vector<bool> processed(this->nvertices);
        std::vector<int> exit_time(this->nvertices);

        DFS(0,
            __time,
            finished,
            discovered, 
            entry_time, 
            processed,
            parent,
            exit_time,
            [](int v) -> void {},
            [&parent, &finished, this](int x, int y) -> void 
            {
                if (parent[y] != x)
                {
                    printf("Cycle from %d to %d: ", y, x);
                    find_path(y, x, parent);
                    finished = true;
                }
            },
            [](int v) -> void {});

    }

    void detectArticulationEdges()
    {
        int __time = 0;
        bool finished = false;
        std::vector<int> reachable_ancestor(this->nvertices);
        std::vector<int> parent(this->nvertices);
        std::vector<int> tree_out_degree(this->nvertices);
        std::vector<int> entry_time(this->nvertices);
        std::vector<bool> discovered(this->nvertices);
        std::vector<bool> processed(this->nvertices);
        std::vector<int> exit_time(this->nvertices);

        DFS(0,
            __time,
            finished,
            discovered, 
            entry_time, 
            processed,
            parent,
            exit_time,
            [&reachable_ancestor](int v) -> void 
            {
                reachable_ancestor[v] = v;
            },
            [&reachable_ancestor, &tree_out_degree, &parent, &entry_time, &discovered, &processed, this](int x, int y) -> void 
            {
                int edge_class;

                edge_class = edge_classification(parent, discovered, processed, entry_time, x, y);

                if (edge_class == TREE)
                    tree_out_degree[x] = tree_out_degree[x] + 1;

                if ((edge_class == BACK) && (parent[x] != y))
                {
                    if (entry_time[y] < entry_time[reachable_ancestor[x]])
                    {
                        reachable_ancestor[x] = y;
                    }
                }
            },
            [&reachable_ancestor, &tree_out_degree, &parent, &entry_time](int v) -> void 
            {
                bool root;
                int time_v;
                int time_parent;

                if (parent[v] == -1)
                {
                    if (tree_out_degree[v] > 1)
                        printf("root articulation vertex: %d \n", v);
                    return;
                }
                
                root = (parent[parent[v]] == -1);

                if (!root)
                {
                    if (reachable_ancestor[v] == parent[v])
                    {
                        printf("parent articulation vertex: %d \n", parent[v]);

                        if (tree_out_degree[v] > 0)
                            printf("bridge articulation vertex: %d \n", v);
                    }
                }

                time_v = entry_time[reachable_ancestor[v]];
                time_parent = entry_time[reachable_ancestor[parent[v]]];

                if (time_v < time_parent)
                    reachable_ancestor[parent[v]] = reachable_ancestor[v];
            });
    }

    void twocolor()
    {
        int i;
        std::vector<bool> discovered(this->nvertices);
        std::vector<bool> processed(this->nvertices);
        std::vector<int> parent(this->nvertices);
        std::vector<int> color(this->nvertices);

        for (i = 0; i < this->nvertices; i++)
        {
            color[i] = UNCOLORED;
        }

        bipartite = true;

        for (i = 0; i < this->nvertices; i++)
        {
            if (!discovered[i])
            {
                color[i] = WHITE;
                BFS(i, discovered, processed, parent, 
                        [](int v) -> void {}, 
                        [&color, this](int x, int y) -> void 
                        {
                            if (color[x] == color[y])
                            {
                                this->bipartite = false;
                                printf("Warning: not bipartite, due to (%d,%d)\n", x, y);
                            }

                            color[y] = complement(color[x]);
                        }, 
                        [](int v) -> void {}); 
            }
        }
    }

    int complement(int color)
    {
        if (color == WHITE)
            return BLACK;

        if (color == BLACK)
            return WHITE;

        return UNCOLORED;
    }

    int connected_components()
    {
        int c;
        int i;
        std::vector<bool> discovered(this->nvertices);
        std::vector<bool> processed(this->nvertices);
        std::vector<int> parent(this->nvertices);

        c = 0;

        for (i = 0; i < nvertices; i++)
        {
            if (!discovered[i])
            {
                c = c + 1;
                //printf("Component %d:", c);
                BFS(i, discovered, processed, parent, [](int v) -> void {}, [](int x, int y) -> void {}, [](int v) -> void {});
            }
        }

        return c;
    }

    void topologicalSort()
    {
        std::stack<int> sorted;

        int __time;
        bool finished;
        std::vector<bool> discovered(this->nvertices);
        std::vector<bool> processed(this->nvertices);
        std::vector<int> parent(this->nvertices);
        std::vector<int> entry_time(this->nvertices);
        std::vector<int> exit_time(this->nvertices);

        for (int i = 1; i < this->nvertices; i++)
        {
            if (!discovered[i])
                DFS(i, 
                        __time,
                        finished,
                        discovered,
                        entry_time,
                        processed,
                        parent,
                        exit_time,
                        [&sorted](int v) -> void {sorted.push(v);}, 
                        [&parent, &discovered, &processed, &entry_time, this](int x, int y) -> void 
                        {
                            if (edge_classification(parent, discovered, processed, entry_time, x,y) == BACK)
                                printf("Warning: directed cycle found, not a DAG\n");
                        }, 
                        [](int v) -> void {});
        }

        while (!sorted.empty())
        {
            printf("%d ", sorted.top());
        }
    }

    Graph transpose()
    {
        Graph graph(this->nvertices);
   
        for (int i = 0; i < this->nvertices; i++)
        {
            std::vector<edgenode> edges = this->edges[i];
            
            for(edgenode edge : edges)
            {
                graph.addEdge(edge.y, i);
            }

        }

        return graph;
    }

    void strong_components()
    {
        Graph graph;

        std::stack<int> s;
        int __time;
        bool finished;
        std::vector<bool> discovered(this->nvertices);
        std::vector<bool> processed(this->nvertices);
        std::vector<int> parent(this->nvertices);
        std::vector<int> entry_time(this->nvertices);
        std::vector<int> exit_time(this->nvertices);

        for (int i = 0; i < this->nvertices; i++)
        {
            if (!discovered[i])
                DFS(i, 
                    __time,
                    finished, 
                    discovered,
                    entry_time,
                    processed,
                    parent,
                    exit_time,
                    [&s](int v) -> void { s.push(v); }, [](int x, int y) -> void {}, [](int v) -> void{});
        }

        graph = this->transpose(); 

        int components_found = 0;
        int v;

        while (!s.empty())
        {
            v = s.top();
            s.pop();
            
            if (!discovered[v])
            {
                components_found++; 
                printf("Component %d:", components_found);
                DFS(v, 
                    __time,
                    finished,
                    discovered,
                    entry_time,
                    processed,
                    parent,
                    exit_time,
                    [](int v) -> void {}, [](int x, int y) -> void {}, [](int v) -> void {});
            }
        }
    }

    int prim(int s)
    {
        std::vector<bool> intree(this->nvertices);
        std::vector<int> distance(this->nvertices);
        std::vector<int> parent(this->nvertices);
        int v; // current vertex to process
        int w; // candidate next vertex
        int dist; // cheapest cost to enlarge tree
        int weight = 0; // tree weight
        
        for (int i = 0; i < this->nvertices; i++)
        {
            intree[i] = false;
            distance[i] = INT_MAX;
            parent[i] = -1;
        }

        distance[s] = 0;
        v = s; 

        while (!intree[v])
        {
            intree[v] = true;

            if (v != s)
            {
                printf("edge (%d %d) in tree \n", parent[v], v);
                weight = weight + dist;
            }

            for (edgenode edge : this->edges[v])
            {
                w = edge.y;

                if ((distance[w] > edge.weight) && (!intree[w]))
                {
                    distance[w] = edge.weight;
                    parent[w] = v;
                }
            }

            dist = INT_MAX;

            for (int i = 0; i < nvertices; i++)
            {
                if ((!intree[i]) && (dist > distance[i]))
                {
                    dist = distance[i];
                    v = i;
                }
            }
        }

        return weight;
    }

    int kruskal()
    {
        UnionFind s(this->nvertices);
        std::vector<edgepair> e;
        int weight = 0;
        for(int i = 0; i < this->nvertices; i++)
        {
            for(int j = 0; j < this->edges[i].size(); j++)
            {
                e.push_back(edgepair(i,this->edges[i][j].y,this->edges[i][j].weight));
            }
        }

        std::sort(e.begin(), e.end(), [](const edgepair& x, const edgepair& y) -> int {return x.weight < y.weight;} );

        for (int i = 0; i < e.size(); i++)
        {
            if (!s.same_component(e[i].x, e[i].y))
            {
                std::cout<<"edge " << e[i].x << "," << e[i].y << " in MST\n";
                weight = weight + e[i].weight;
                s.union_sets(e[i].x, e[i].y);
            }
        }

        return weight;
    }

    int dijkstra(int start)
    {
        std::vector<bool> intree(this->nvertices);
        std::vector<int> distance(this->nvertices);
        std::vector<int> parent(this->nvertices);
        int v; // current vertex to process
        int w; // candidate next vertex 
        int dist; // cheapest cost to enlarge tree
        int weight = 0; // tree weight
        
        for (int i = 0; i < this->nvertices; i++)
        {
            intree[i] = false;
            distance[i] = INT_MAX;
            parent[i] = -1;
        }

        distance[start] = 0;
        v = start;

        while(!intree[v])
        {
            intree[v] = true;
            if (v != start)
            {
                printf("edge (%d,%d) in tree \n", parent[v], v);
                weight = weight + dist;
            }

            for (edgenode edge : this->edges[v])
            {
                w = edge.y;
                if (distance[w] > (distance[v] + edge.weight))
                {
                    distance[w] = distance[v] + edge.weight;
                    parent[w] = v;
                }
            }

            dist = INT_MAX;

            for (int i = 0; i <= this->nvertices; i++)
            {
                if ((!intree[i]) && (dist > distance[i]))
                {
                    dist = distance[i];
                    v = i;
                }
            }
        }

        return weight;
    }

    bool valid_edge(edgenode e)
    {
        return e.residual > 0;
    }

    edgenode* find_edge(int x, int y)
    {
        for (int i = 0; i < this->edges[x].size(); i++)
        {
            if (this->edges[x][i].y == y)
                return &this->edges[x][i];
        }

        return nullptr;
    }

    void augment_path(std::vector<int>& parent, int start, int end, int volume)
    {
        edgenode* e;

        if (start == end)
            return; 

        e = find_edge(parent[end], end);
        e->flow += volume;
        e->residual -= volume;

        e = find_edge(end, parent[end]);
        e->residual += volume;

        augment_path(parent, start, parent[end], volume);
    }

    int path_volume(std::vector<int>& parent, int start, int end)
    {
        edgenode* e;
        if (parent[end] == -1)
            return 0;

        e = find_edge(parent[end], end);

        if (start == parent[end])
            return e->residual;
        else 
            return std::min(path_volume(parent, start, parent[end]), e->residual);
    }

    void add_residual_edges() {
        // Iterate over all edges in the adjacency list
        for (int u = 0; u < this->nvertices; ++u) {
            for (auto& edge : this->edges[u]) {
                int v = edge.y; // Target vertex
                int capacity = edge.weight; // Capacity of the edge

                // Check if a residual edge from v to u already exists
                bool residual_exists = false;
                for (auto& reverse_edge : this->edges[v]) {
                    if (reverse_edge.y == u) {
                        residual_exists = true;
                        break;
                    }
                }

                // If no residual edge exists, add one with a capacity of 0
                if (!residual_exists) {
                    this->edges[v].emplace_back(edgenode{u, 0});
                }
            }
        }
    }
    
    int netflow(int source, int sink)
    {
        int total_flow = 0; // Accumulate the total flow here
        int volume;
        std::vector<int> parent(this->nvertices);
        std::vector<bool> processed(this->nvertices);
        std::vector<bool> discovered(this->nvertices);

        // Add residual edges to the graph
        add_residual_edges();

        // Find the initial augmenting path
        std::fill(discovered.begin(), discovered.end(), false);
        std::fill(processed.begin(), processed.end(), false);
        std::fill(parent.begin(), parent.end(), -1);
        BFS(source, discovered, processed, parent, [](int v) -> void {}, [&](int x, int y) -> void 
                {
edgenode* e = find_edge(x, y);
                        if (e && e->residual > 0) {
                            parent[y] = x; 
                        }
                }, [](int v) -> void {});
        
        // Compute the volume of the augmenting path
        volume = path_volume(parent, source, sink);
        // While there is an augmenting path, push flow and update the residual graph
        while (volume > 0)
        {
            total_flow += volume; // Add the flow of this path to the total flow
            augment_path(parent, source, sink, volume); // Update the flow along the path

            // Find the next augmenting path
            std::fill(discovered.begin(), discovered.end(), false);
            std::fill(processed.begin(), processed.end(), false);
            std::fill(parent.begin(), parent.end(), -1);
            BFS(source, discovered, processed, parent, [](int v) -> void {}, [&](int x, int y) -> void 
                    {
                        edgenode* e = find_edge(x, y);
                        if (e && e->residual > 0) {
                            parent[y] = x; 
                        }
                    }, [](int v) -> void {});

            // Compute the volume of the new augmenting path
            volume = path_volume(parent, source, sink);
            std::cout<<volume<<" ";
        }

        return total_flow; // Return the total flow computed
    }
    
    void floyd (adjacency_matrix* g)
    {
        int i, j;
        int k;
        int through_k;

        for (k = 0; k < g->nvertices; k++)
        {
            for (i = 0; i < g->nvertices; i++)
            {
                for (j = 0; j < g->nvertices; j++)
                {
                    through_k = g->weight[i][k] + g->weight[k][j];
                    if (through_k < g->weight[i][j])
                        g->weight[i][j] = through_k;
                }
            }
        }
    }

    std::vector<int> bellmanFord(int n, std::vector<std::vector<int>>& edges, int src) {
        std::vector<int> dist(n, INT_MAX);
        dist[src] = 0;
        
        for(int i=0; i<n-1; ++i)
            for(auto& e : edges)
                if(dist[e[0]] != INT_MAX && dist[e[1]] > dist[e[0]] + e[2])
                    dist[e[1]] = dist[e[0]] + e[2];
        
        for(auto& e : edges)
            if(dist[e[0]] != INT_MAX && dist[e[1]] > dist[e[0]] + e[2])
                return {};
        
        return dist;
    }

    int fordFulkerson(std::vector<std::vector<int>> graph, int s, int t) {
        std::vector<std::vector<int>> rGraph = graph;
        std::vector<int> parent(graph.size());
        int max_flow = 0;
        while(int flow = bfs(rGraph, s, t, parent)) {
            max_flow += flow;
            for(int v=t; v!=s; v=parent[v]) {
                int u = parent[v];
                rGraph[u][v] -= flow;
                rGraph[v][u] += flow;
            }
        }

        return max_flow;
    }
    bool havelHakimi(std::vector<int> degrees) {
        while(true) {
            sort(degrees.rbegin(), degrees.rend());
            
            while(!degrees.empty() && degrees.back() == 0) 
                degrees.pop_back();
            
            if(degrees.empty()) 
                return true;
            
            int n = degrees[0];
            degrees.erase(degrees.begin());
            
            if(n > degrees.size()) 
                return false;
            
            for(int i=0; i<n; ++i) {
                degrees[i]--;
                if(degrees[i] < 0) return false;
            }
        }
    }

    std::vector<int> hierholzer(std::vector<std::vector<int>>& adj) {
        std::vector<int> path;
        std::stack<int> st;
        st.push(0);
        while(!st.empty()) {
            int u = st.top();
            if(adj[u].empty()) {
                path.push_back(u);
                st.pop();
            } else {
                int v = adj[u].back();
                adj[u].pop_back();
                st.push(v);
            }
        }
        reverse(path.begin(), path.end());
        return path;
    }

    std::vector<std::vector<int>> kosaraju(std::vector<std::vector<int>>& adj) {
        int n = adj.size();
        std::vector<int> order;
        std::vector<bool> visited(n, false);
        std::function<void(int)> dfs1 = [&](int u) {
            visited[u] = true;
            for(int v : adj[u])
                if(!visited[v]) dfs1(v);
            order.push_back(u);
        };
        for(int i=0; i<n; ++i)
            if(!visited[i]) dfs1(i);
        
        std::vector<std::vector<int>> transpose(n);
        for(int u=0; u<n; ++u)
            for(int v : adj[u])
                transpose[v].push_back(u);
        
        std::vector<std::vector<int>> scc;
        fill(visited.begin(), visited.end(), false);
        reverse(order.begin(), order.end());
        std::function<void(int, std::vector<int>&)> dfs2 = [&](int u, std::vector<int>& component) {
            component.push_back(u);
            visited[u] = true;
            for(int v : transpose[u])
                if(!visited[v]) dfs2(v, component);
        };
        for(int u : order)
            if(!visited[u]) {
                scc.push_back({});
                dfs2(u, scc.back());
            }
        return scc;
    }
};
int tsp(std::vector<std::vector<int>>& dist) {
    int n = dist.size();
    std::vector<std::vector<int>> dp(1<<n, std::vector<int>(n, INT_MAX));
    dp[1][0] = 0;
    for(int mask=1; mask<(1<<n); ++mask)
        for(int u=0; u<n; ++u) if(mask & (1<<u))
            for(int v=0; v<n; ++v) if(!(mask & (1<<v)) && dist[u][v] != INT_MAX)
                if(dp[mask][u] != INT_MAX)
                    dp[mask|(1<<v)][v] = std::min(dp[mask|(1<<v)][v], dp[mask][u] + dist[u][v]);
    int res = INT_MAX;
    for(int u=0; u<n; ++u)
        if(dp[(1<<n)-1][u] != INT_MAX && dist[u][0] != INT_MAX)
            res = std::min(res, dp[(1<<n)-1][u] + dist[u][0]);
    return res;
}
std::vector<int> kmp(const std::string& pattern) {
    int m = pattern.size();
    std::vector<int> lps(m, 0);
    for(int i=1, len=0; i<m; ) {
        if(pattern[i] == pattern[len]) {
            lps[i++] = ++len;
        } else {
            if(len) len = lps[len-1];
            else lps[i++] = 0;
        }
    }
    return lps;
}

std::vector<int> kmpSearch(const std::string& text, const std::string& pattern) {
    std::vector<int> lps = kmp(pattern);
    std::vector<int> matches;
    int n = text.size(), m = pattern.size();
    for(int i=0, j=0; i<n; ) {
        if(pattern[j] == text[i]) {
            i++; j++;
        }
        if(j == m) {
            matches.push_back(i-j);
            j = lps[j-1];
        } else if(i < n && pattern[j] != text[i]) {
            j ? j = lps[j-1] : i++;
        }
    }
    return matches;
}

int levenshtein(const std::string& a, const std::string& b) {
    int m = a.size(), n = b.size();
    std::vector<std::vector<int>> dp(m+1, std::vector<int>(n+1, 0));
    for(int i=0; i<=m; ++i) dp[i][0] = i;
    for(int j=0; j<=n; ++j) dp[0][j] = j;
    for(int i=1; i<=m; ++i)
        for(int j=1; j<=n; ++j)
            dp[i][j] = std::min({dp[i-1][j]+1, dp[i][j-1]+1, 
                          dp[i-1][j-1] + (a[i-1] != b[j-1])});
    return dp[m][n];
}

std::ifstream in("maxflow.in");

void getIntersction(int& __max, std::vector<int>& level, std::vector<int>& cost, std::vector<int>& parent, int x, int y)
{
    if (level[x] > level[y]) 
        std::swap(x, y);

    while (level[x] < level[y])
    {
        __max = std::max(__max, cost[y]);
        y = parent[y];
    }

    while (x != y)
    {
        __max = std::max(__max, std::max(cost[x], cost[y]));
        x = parent[x];
        y = parent[y];
    }
}

int main()
{
    int n, m;
    in>>n;
    in>>m;
    Graph g(n, true, true);
    for (int i = 0; i < m; i++)
    {
        int x, y, z;
        in>>x>>y>>z;
        g.addEdge(x-1,y-1,z);
    }

    std::cout<<g.fordFulkerson(g.edges, 0, n - 1);
}
