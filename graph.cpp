#include <iostream>
#include <vector>
#include <stdexcept>
#include <queue>
#include <functional>
#include <algorithm>
#include <stack>
#include <climits>

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
    edgenode() : y(0), weight(0) {}
    edgenode(int y, int weight) : y(y), weight(weight) {}
};

struct edgepair 
{
    int x;
    int y;
    int weight;
};

class UnionFind 
{
    private:
        std::vector<int> p;
        std::vector<int> size;
        int n;
    public:
        UnionFind()
        {
        }
};

class Graph 
{
private:
    bool weighted;
    bool directed;
    bool finished;
    bool bipartite; 

    int nvertices;
    int __time;
    
    /* these may be used by various algorithms */
    std::vector<int> parent;
    std::vector<int> processed;
    std::vector<int> discovered;
    std::vector<int> entry_time;
    std::vector<int> exit_time;
    std::vector<bool> visited;
    std::vector<int> reachable_ancestor;
    std::vector<int> tree_out_degree;
    std::vector<int> color;

    std::vector<std::vector<edgenode>> edges;
public:
    Graph()
    {
    }

    Graph(int nvertices, bool weighted = false)
    {
        this->nvertices = nvertices;
        this->weighted = weighted;
    }
    
    Graph(const Graph& other)
    {
        this->nvertices = other.nvertices;
        this->edges = other.edges;
        this->weighted = other.weighted;
    }

    void newVertex()
    {
        nvertices++;
    }
    
    void addEdge(int v, edgenode e, int weight = 0)
    {
        edges[v].push_back(e);
        edges[e.y].push_back(edgenode(v, weight));
    }

    void addEdge(int vertex1, int vertex2, int weight = 0)
    {
        if(vertex1 >= nvertices)
           throw std::out_of_range("Invalid vertex1 index"); 

        if(vertex2 >= nvertices)
           throw std::out_of_range("Invalid vertex2 index"); 

        edges[vertex1].push_back(edgenode(vertex2, weight));
        edges[vertex2].push_back(edgenode(vertex1, weight));
    }

    void process_articulation_vertex_early(int v) 
    {
        reachable_ancestor[v] = v;
    }

    void process_articulation_edge(int x, int y)
    {
        int edge_class; 

        edge_class = edge_classification(x, y);

        if (edge_class == TREE)
            tree_out_degree[x] = tree_out_degree[x] + 1;
        
        if ((edge_class == BACK) && (parent[x] != y)) 
        {
            if (entry_time[y] < entry_time[reachable_ancestor[x]])
            {
                reachable_ancestor[x] = y;
            }
        }
    }

    void process_articulation_vertex_late(int v) 
    {
        bool root;
        int time_v;
        int time_parent;

        if (parent[v] == -1)
        {
            if (tree_out_degree[v] > 1)
            {
                printf("root articulation vertex: %d \n", v);
            }
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
    }

    void BFS(int s, 
            const std::function<void(int)>& process_vertex_early, 
            const std::function<void(int, int)>& process_edge, 
            const std::function<void(int)>& process_vertex_late) 
    {
        std::priority_queue<int> pq;
        bool visited[nvertices];
        pq.push(s);
        
        while(!pq.empty())
        {
            int vertex = pq.top(); 
            pq.pop();
            process_vertex_early(vertex);
            if(!visited[vertex])
            {
                visited[vertex] = true;
                for(edgenode edge : edges[vertex])
                {
                    process_edge(vertex, edge.y);
                    pq.push(edge.y);
                }
            }

            process_vertex_late(vertex);
        }
    }

    int edge_classification(int x, int y)
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

    void edge_detect(int x, int y)
    {
        if (parent[y] != x) 
        {
            printf("Cycle from %d to %d: ", y, x);
            find_path(y, x, parent); 
            finished = true;
        }
    }

    void DFS(int s, 
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
                DFS(edge.y, process_vertex_early, process_edge, process_vertex_late);
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

    void twocolor()
    {
        int i;

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
                BFS(i, [](int v) -> void {}, [](int x, int y) -> void {}, [](int v) -> void {}); 
            }
        }
    }

    void bipartite_process_edge(int x, int y)
    {
        if (color[x] == color[y])
        {
            bipartite = false;
            printf("Warning: not bipartite, due to (%d,%d)\n", x, y);
        }

        color[y] = complement(color[x]);
    }

    int complement(int color)
    {
        if (color == WHITE)
            return BLACK;

        if (color == BLACK)
            return WHITE;

        return UNCOLORED;
    }

    void connected_components()
    {
        int c;
        int i;

        c = 0;

        for (i = 0; i < nvertices; i++)
        {
            if (!discovered[i])
            {
                c = c + i;
                printf("Component %d:", c);
                BFS(i, [](int v) -> void {}, [](int x, int y) -> void {}, [](int v) -> void {});
            }
        }
    }

    void topologicalSort()
    {
        std::stack<int> sorted;

        for (int i = 1; i < this->nvertices; i++)
        {
            if (!discovered[i])
                DFS(i, [](int v) -> void {}, [](int x, int y) -> void {}, [](int v) -> void {});
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

        for (int i = 0; i < this->nvertices; i++)
        {
            if (!discovered[i])
                DFS(i, [&s](int v) -> void { s.push(v); }, [](int x, int y) -> void {}, [](int v) -> void{});
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
                DFS(v, [](int v) -> void {}, [](int x, int y) -> void {}, [](int v) -> void {});
            }
        }
    }

    int prim(int s)
    {
        std::vector<bool> intree(this->nvertices);
        std::vector<int> distance(this->nvertices);
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
        unionFind s;
        std::vector<edgepair> e(this->nvertices);
        int weight = 0;

        toEdgeArray(e);
        std::sort(e.begin(), e.end(), []() -> int {});

        for (int i = 0; i < this->edges.size(); i++)
        {
            if (!same_component(s, e[i].x, e[i].y))
            {
                printf("edge (%d, %d) in MST\n", e[i].x, e[i].y);
                weight = weight + e[i].weight;
                union_sets(s, e[i].y, e[i].y);
            }
        }

        return weight;
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
};
