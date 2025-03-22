#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <time.h>
#include <limits>
#include <omp.h>

using namespace std;

int selA = 5;
int V = 10;

class Graph {
    int V;
    vector<vector<pair<int, int>>> adjList;
    vector<vector<int>> edgeCosts;

public:
    explicit Graph(int V) {
        this->V = V;
        adjList.resize(V);
        edgeCosts.resize(V, vector<int>(V, 0));
    }

    void addEdge(int u, int v, int w) {
        adjList[u].emplace_back(v, w);
        edgeCosts[u][v] = w;
    }

    void printGraph() {
        for (int i = 0; i < V; ++i) {
            cout << "Vertex " << i << ":";
            for (auto edge : adjList[i]) {
                cout << " -> (" << edge.first << ", weight: " << edge.second << ")";
            }
            cout << endl;
        }
    }

    [[nodiscard]] const vector<pair<int, int>>& getEdges(int u) const {
        return adjList[u];
    }

    [[nodiscard]] int getEdgeCost(int u, int v) const {
        return edgeCosts[u][v];
    }

    [[nodiscard]] int getVertexCount() const  {
        return V;
    }
};

void loadFile(const string& fileName, Graph &G) {
    ifstream infile(fileName);
    if (!infile) {
        cerr << "Error opening file: " << fileName << endl;
        exit(1);
    }

    vector<vector<int>> matrix;
    string line;

    getline(infile, line);

    while(getline(infile, line)) {
        stringstream ss(line);
        vector<int> row;
        int value;
        while(ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }

    int size = matrix.size();
    for(int i=0; i<size; ++i) {
        for(int j=0; j<size; ++j) {
            if(matrix[i][j] != 0 && i < j) {
                G.addEdge(i, j, matrix[i][j]);
            }
        }
    }
}

int calculateCost(const Graph &G, const vector<int> &flags, int node, int flag) {
    int cost = 0;
    const vector<pair<int,int>>& neighbors = G.getEdges(node);

    for (const auto &neighbor : neighbors) {
        int neighborNode = neighbor.first;
        int weight = neighbor.second;
        if (flags[neighborNode] != -1 && flags[neighborNode] != flag) {
            cost += weight;
        }
    }
    for (int i = 0; i < node; ++i) {
        if (flags[i] != -1 && flags[i] != flag) {
            cost += G.getEdgeCost(i, node);
        }
    }
    return cost;
}

void branchAndBoundSerial(const Graph &G, vector<int> &flags, int index, int countA, int currentCost, int &bestCost)
{
    if (currentCost >= bestCost) return;
    if (countA > selA || countA + (V - index) < selA) return;

    if (index == V) {
        if (currentCost < bestCost) {
            bestCost = currentCost;
        }
        return;
    }

    flags[index] = 0;
    int costIncreaseA = calculateCost(G, flags, index, 0);
    branchAndBoundSerial(G, flags, index + 1, countA + 1,
                         currentCost + costIncreaseA, bestCost);

    flags[index] = 1;
    int costIncreaseB = calculateCost(G, flags, index, 1);
    branchAndBoundSerial(G, flags, index + 1, countA,
                         currentCost + costIncreaseB, bestCost);

    flags[index] = -1;
}


static const int MAX_DEPTH_PARALLEL = 8;

void branchAndBoundParallel(const Graph &G, vector<int> flags,
                            int index, int countA, int currentCost, int &globalMinCost, int depth) {
    int localMinCost;
    #pragma omp atomic read
    localMinCost = globalMinCost;

    if (currentCost >= localMinCost) return;
    if (countA > selA || countA + (V - index) < selA) return;

    if (index == V) {
        #pragma omp critical
        {
            if (currentCost < globalMinCost) {
                globalMinCost = currentCost;
            }
        }
        return;
    }

    if (depth < MAX_DEPTH_PARALLEL) {
        flags[index] = 0;
        int costA = calculateCost(G, flags, index, 0);
        #pragma omp task default(none) \
                         shared(G, globalMinCost) \
                         firstprivate(flags, index, countA, currentCost, depth, costA)
        {
            branchAndBoundParallel(G, flags, index + 1,
                                   countA + 1,
                                   currentCost + costA,
                                   globalMinCost,
                                   depth + 1);
        }

        flags[index] = 1;
        int costB = calculateCost(G, flags, index, 1);
        #pragma omp task default(none) \
                         shared(G, globalMinCost) \
                         firstprivate(flags, index, countA, currentCost, depth, costB)
        {
            branchAndBoundParallel(G, flags, index + 1,
                                   countA,
                                   currentCost + costB,
                                   globalMinCost,
                                   depth + 1);
        }
    }
    else {
        flags[index] = 0;
        int costA = calculateCost(G, flags, index, 0);
        branchAndBoundSerial(G, flags, index + 1, countA + 1, currentCost + costA, globalMinCost);

        flags[index] = 1;
        int costB = calculateCost(G, flags, index, 1);
        branchAndBoundSerial(G, flags, index + 1, countA, currentCost + costB, globalMinCost);
    }
    #pragma omp taskwait
}


int main() {
    V = 30;
    selA = 15;

    Graph g(V);
    loadFile("/Users/surya_v/CLionProjects/untitled/graf_30_20.txt", g);

    g.printGraph();

    vector<int> flags(V, -1);
    int globalMinCost = numeric_limits<int>::max();

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            branchAndBoundParallel(g, flags, 0, 0, 0, globalMinCost,0);
        }
    }

    cout << "Minimum cut cost: " << globalMinCost << endl;
    double stop = omp_get_wtime();
    double elapsed = stop - start;
    printf("\nTime elapsed: %.5f\n", elapsed);

    return 0;
}
