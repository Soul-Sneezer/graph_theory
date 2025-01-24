#include <iostream>
#include <vector>
#include <stdexcept>
#include <queue>
#include <functional>

const int TREE = 1;
const int BACK = 2;
const int FORWARD = 3;
const int CROSS = 4;

struct edgenode 
{
    int y;
    int weight;
    edgenode() : y(0), weight(0) {}
    edgenode(int y, int weight) : y(y), weight(weight) {}
};

class Graph 
{
private:
    bool weighted;
    int nvertices;
    std::vector<int> parent;
    std::vector<int> processed;
    std::vector<int> entry_time;
    std::vector<bool> visited;
    std::vector<std::vector<edgenode>> edges;

    void process_vertex_early(int vertex);
    void process_vertex_late(int vertex);
    void process_edge(edgenode edge);
public:
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

    void addEdge(int vertex1, int vertex2, int weight = 0)
    {
        if(vertex1 >= nvertices)
           throw std::out_of_range("Invalid vertex1 index"); 

        if(vertex2 >= nvertices)
           throw std::out_of_range("Invalid vertex2 index"); 

        edges[vertex1].push_back(edgenode(vertex2, weight));
        edges[vertex2].push_back(edgenode(vertex1, weight));
    }

    void BFS(int s) 
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
                    process_edge(edge);
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

    void DFS(int s, const std::function<int(int)>& process_vertex_early, const std::function<int(int>& process_edge, const std::function<int(int)>* process_vertex_late)
    {
        visited[s] = true;
        process_vertex_early(s);
        for(edgenode edge : edges[s])
        {
            process_edge(edge);
            if (!visited[edge.y])
                DFS(edge.y);
        }
        process_vertex_late(s);
    }
};
