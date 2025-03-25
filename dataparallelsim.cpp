#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <queue>
#include <sstream>
#include <limits>
#include <omp.h>

using namespace std;

int V = 30;
int selA = 15;
int ENOUGH_STATES = 10000;


static int  g_bestCost  = numeric_limits<int>::max();
static vector<int> g_bestFlags;

class Graph {
    int V;
    vector<vector<pair<int,int>>> adjList;
    vector<vector<int>> edgeCosts;

public:
    explicit Graph(int V) : V(V) {
        adjList.resize(V);
        edgeCosts.resize(V, vector<int>(V, 0));
    }

    void addEdge(int u, int v, int w) {
        adjList[u].emplace_back(v, w);
        edgeCosts[u][v] = w;
    }
    const vector<pair<int, int>>& getEdges(int u) const {
        return adjList[u];
    }
    int getEdgeCost(int u, int v) const {
        return edgeCosts[u][v];
    }
    int getVertexCount() const {
        return V;
    }
    void printGraph() {
        for (int i = 0; i < V; ++i) {
            cout << "Vertex " << i << ":";
            for (auto &edge : adjList[i]) {
                cout << " -> (" << edge.first << ", weight: " << edge.second << ")";
            }
            cout << endl;
        }
    }
};

struct BnBState {
    vector<int> flags;
    int index;
    int countA;
    int cost;
};

void loadFile(const string& fileName, Graph &G) {
    ifstream infile(fileName);
    if (!infile) {
        cerr << "Error opening file: " << fileName << endl;
        exit(1);
    }

    string line;
    getline(infile, line);

    vector<vector<int>> matrix;
    while (getline(infile, line)) {
        stringstream ss(line);
        vector<int> row;
        int value;
        while (ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }

    int size = (int)matrix.size();
    for(int i=0; i<size; ++i) {
        for(int j=0; j<size; ++j) {
            if(matrix[i][j] != 0 && i<j) {
                G.addEdge(i, j, matrix[i][j]);
            }
        }
    }
}

int calculateCost(const Graph &G, const vector<int> &flags, int node, int flag) {
    // Same as task parallel
    int cost = 0;
    const auto &neighbors = G.getEdges(node);
    for (auto &nbr : neighbors) {
        int nnode  = nbr.first;
        int weight = nbr.second;
        if (flags[nnode] != -1 && flags[nnode] != flag) {
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


void resAlmostSeqSubTree(const Graph &G, vector<int> &flags, int index, int countA, int currentCost)
{
    // almost same as task parallel
    if (currentCost >= g_bestCost) return;
    if (countA > selA || (countA + (V - index)) < selA) return;

    if (index == V) {
        #pragma omp critical
        {
            if (currentCost < g_bestCost) {
                g_bestCost  = currentCost;
                g_bestFlags = flags;
            }
        }
        return;
    }

    flags[index] = 0;
    {
        int cInc = calculateCost(G, flags, index, 0);
        resAlmostSeqSubTree(G, flags, index + 1, countA + 1, currentCost + cInc);
    }

    flags[index] = 1;
    {
        int cInc = calculateCost(G, flags, index, 1);
        resAlmostSeqSubTree(G, flags, index + 1, countA, currentCost + cInc);
    }

    flags[index] = -1;
}

vector<BnBState> buildPartialStates(const Graph &G, int enoughStates, int maxDepth) {
    vector<BnBState> result;
    queue<BnBState> Q;

    BnBState init;
    init.flags  = vector<int>(V, -1);
    init.index  = 0;
    init.countA = 0;
    init.cost   = 0;
    Q.push(init);

    while (!Q.empty()) {
        BnBState s = Q.front();
        Q.pop();

        if (s.index >= maxDepth || s.index == V) {
            result.push_back(s);
        }
        else {
            {
                BnBState s0 = s;
                s0.flags[s0.index] = 0;
                int cInc = calculateCost(G, s0.flags, s0.index, 0);
                s0.cost   += cInc;
                s0.countA += 1;
                s0.index  += 1;
                // Prune
                if (s0.cost < g_bestCost && s0.countA <= selA) {
                    Q.push(s0);
                }
            }

            {
                BnBState s1 = s;
                s1.flags[s1.index] = 1;
                int cInc = calculateCost(G, s1.flags, s1.index, 1);
                s1.cost  += cInc;
                s1.index += 1;
                // Prune
                if (s1.cost < g_bestCost) {
                    Q.push(s1);
                }
            }
        }

        if ((int)result.size() >= enoughStates) break;
    }

    return result;
}

int main() {
    Graph g(V);
    loadFile("data/graf_30_20.txt", g);

    g.printGraph();

    g_bestFlags.resize(V, -1);

    double start = omp_get_wtime();

    // BFS
    int BFS_DEPTH = 8;
    vector<BnBState> partialStates = buildPartialStates(g, ENOUGH_STATES, BFS_DEPTH);

    cout << "Generated " << partialStates.size()
         << " partial states from BFS.\n";

    // recursion on each partial state
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)partialStates.size(); i++) {
        BnBState st = partialStates[i];
        // weird result one time, so i created a local variable instead of directly passing st.flags
        vector<int> localFlags = st.flags;
        resAlmostSeqSubTree(g, localFlags, st.index, st.countA, st.cost);
    }

    double stop = omp_get_wtime();

    cout << "\nMinimum cut cost: " << g_bestCost << endl;
    cout << "\nTime: " << (stop - start) << " seconds\n";

    return 0;
}
