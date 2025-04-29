#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <time.h>
#include <limits>

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
        while(ss >> value)
            row.push_back(value);
        matrix.push_back(row);
    }

    int V = matrix.size();
    for(int i=0; i<V; ++i) {
        for(int j=0; j<V; ++j) {
            if(matrix[i][j] != 0 && i<j) {
                G.addEdge(i, j, matrix[i][j]);
            }
        }
    }
}

int calculateCost(const Graph &G, const vector<int> &flags, int node, int flag) {
    int cost = 0;
    const vector<pair<int, int>>& neighbors = G.getEdges(node);

    for (const auto& neighbor : neighbors) {
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

void branchAndBound(const Graph &G, vector<int> &flags, int index, int countA, int currentCost, int &minCost) {
    if (currentCost >= minCost) {
        return;
    }

    if (countA > selA || countA + (V - index) < selA) {
        return;
    }

    if (index == V) {
        minCost = min(minCost, currentCost);
        return;
    }

    flags[index] = 0;
    int costIncreaseA = calculateCost(G, flags, index, 0);
    branchAndBound(G, flags, index + 1, countA + 1, currentCost + costIncreaseA, minCost);

    flags[index] = 1;
    int costIncreaseB = calculateCost(G, flags, index, 1);
    branchAndBound(G, flags, index + 1, countA, currentCost + costIncreaseB, minCost);

    flags[index] = -1;
}

int main(int argc, char *argv[]) {
    V = std::stoi(argv[1]);
    selA = std::stoi(argv[2]);
    Graph g(V);
    vector<int> flags(V, -1);
    int minCost = numeric_limits<int>::max();
    std::string filename = argv[3];
    loadFile(filename, g);

    g.printGraph();

    clock_t start = clock();

    branchAndBound(g, flags, 0, 0, 0, minCost);
    cout << "Minimum cut cost: " << minCost << endl;

    clock_t stop = clock();
    double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5f\n", elapsed);

    return 0;
}
