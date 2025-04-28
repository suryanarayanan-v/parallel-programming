#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <queue>
#include <sstream>
#include <limits>
#include <chrono> // Using chrono for timing example
#include <mpi.h>
#include <cstring> // For memcpy
#include <cassert> // For assertions

using namespace std;

int V = 30;
int selA = 15;
int ENOUGH_STATES = 10000;

int g_bestCost = numeric_limits<int>::max();
vector<int> g_bestFlags;

const int TAG_GRAPH_ADJ_SIZE = 10;
const int TAG_GRAPH_ADJ_DATA = 11;
const int TAG_GRAPH_COSTS = 12;
const int TAG_WORK_UNIT = 13;
const int TAG_RESULT = 14;
const int TAG_TERMINATE = 15;
const int TAG_BEST_COST_UPDATE = 16;
const int TAG_REQUEST_FLAGS = 17;
const int TAG_FINAL_FLAGS = 18;


class Graph {
    int V_local;
    vector<vector<pair<int,int>>> adjList;
    vector<vector<int>> edgeCosts;

public:
    Graph() : V_local(0) {}

    explicit Graph(int v) : V_local(v) {
        adjList.resize(V_local);
        edgeCosts.resize(V_local, vector<int>(V_local, 0));
    }

    void setAdjList(const vector<vector<pair<int,int>>>& newAdjList) {
        adjList = newAdjList;
    }

    void setEdgeCosts(const vector<vector<int>>& newEdgeCosts) {
        edgeCosts = newEdgeCosts;
    }

    void addEdge(int u, int v, int w) {
         if (u < 0 || u >= V_local || v < 0 || v >= V_local) return;
        adjList[u].emplace_back(v, w);
        edgeCosts[u][v] = w;
    }

    const vector<pair<int, int>>& getEdges(int u) const {
         if (u < 0 || u >= V_local) {
            static const vector<pair<int, int>> empty_vec;
            return empty_vec;
         }
        return adjList[u];
    }

    int getEdgeCost(int u, int v) const {
        if (u < 0 || u >= V_local || v < 0 || v >= V_local) return 0;
        return edgeCosts[u][v];
    }

    int getVertexCount() const {
        return V_local;
    }

    void printGraph() {
        cout << "--- Graph (V=" << V_local << ") ---" << endl;
        for (int i = 0; i < V_local; ++i) {
            cout << "Vertex " << i << ":";
            for (auto &edge : adjList[i]) {
                cout << " -> (" << edge.first << ", weight: " << edge.second << ")";
            }
            cout << endl;
        }
        cout << "--------------------" << endl;
    }

    const vector<vector<pair<int,int>>>& getAdjList() const { return adjList; }
    const vector<vector<int>>& getEdgeCostsMatrix() const { return edgeCosts; }
};

struct BnBState {
    vector<int> flags;
    int index;
    int countA;
    int cost;

    BnBState() : index(0), countA(0), cost(0) {}

    BnBState(int v_size) : flags(v_size, -1), index(0), countA(0), cost(0) {}
};

void loadFile(const string& fileName, Graph &G) {
    ifstream infile(fileName);
    if (!infile) {
        cerr << "MPI Rank " << []() { int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }() << ": Error opening file: " << fileName << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
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
        if (!row.empty()) {
             matrix.push_back(row);
        }
    }
    infile.close();

    int size = (int)matrix.size();
    if (size != G.getVertexCount()) {
        cerr << "MPI Rank " << []() { int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }()
             << ": Error: Matrix size (" << size << ") read from file '" << fileName
             << "' does not match Graph size (" << G.getVertexCount() << ")." << endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }


    for(int i=0; i<size; ++i) {
        if (matrix[i].size() != (size_t)size) {
             cerr << "MPI Rank " << []() { int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }()
                  << ": Error: Row " << i << " in matrix from file '" << fileName
                  << "' has incorrect size (" << matrix[i].size() << "), expected " << size << "." << endl;
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        for(int j=0; j<size; ++j) {
            if(matrix[i][j] != 0 && i<j) {
                G.addEdge(i, j, matrix[i][j]);
            }
        }
    }
}


int calculateCost(const Graph &G, const vector<int> &flags, int node, int flag) {
    int cost = 0;
    const auto &neighbors = G.getEdges(node);
    for (auto &nbr : neighbors) {
        int nnode  = nbr.first;
        int weight = nbr.second;
        if (nnode < 0 || nnode >= (int)flags.size()) continue;
        if (flags[nnode] != -1 && flags[nnode] != flag) {
            cost += weight;
        }
    }
    for (int i = 0; i < node; ++i) {
         if (i < 0 || i >= (int)flags.size()) continue;
        if (flags[i] != -1 && flags[i] != flag) {
            cost += G.getEdgeCost(i, node);
        }
    }
    return cost;
}

void sendState(const BnBState& state, int dest, int tag, MPI_Comm comm) {
    int v_size = state.flags.size();
    MPI_Send(&state.index, 1, MPI_INT, dest, tag, comm);
    MPI_Send(&state.countA, 1, MPI_INT, dest, tag, comm);
    MPI_Send(&state.cost, 1, MPI_INT, dest, tag, comm);
    MPI_Send(state.flags.data(), v_size, MPI_INT, dest, tag, comm);
}

void recvState(BnBState& state, int source, int tag, MPI_Comm comm, MPI_Status* status) {
    int v_size = state.flags.size();
    assert(v_size > 0 && "recvState: state.flags must be pre-sized.");
    MPI_Recv(&state.index, 1, MPI_INT, source, tag, comm, status);
    MPI_Recv(&state.countA, 1, MPI_INT, source, tag, comm, status);
    MPI_Recv(&state.cost, 1, MPI_INT, source, tag, comm, status);
    MPI_Recv(state.flags.data(), v_size, MPI_INT, source, tag, comm, status);
}

int resAlmostSeqSubTreeMPI(const Graph &G, vector<int> &flags, int index, int countA, int currentCost,
                           const int& currentGlobalBestCost,
                           vector<int>& bestFlags_local)
{
    if (currentCost >= currentGlobalBestCost) return numeric_limits<int>::max();
    if (countA > selA || (countA + (V - index)) < selA) return numeric_limits<int>::max();

    if (index == V) {
        bestFlags_local = flags;
        return currentCost;
    }

    int bestCostFound = numeric_limits<int>::max();
    vector<int> currentBestFlags = flags;

    flags[index] = 0;
    {
        int cInc = calculateCost(G, flags, index, 0);
        if (currentCost + cInc < currentGlobalBestCost && (countA + 1) <= selA) {
            vector<int> flagsFromBranch0(V);
            int cost0 = resAlmostSeqSubTreeMPI(G, flags, index + 1, countA + 1, currentCost + cInc,
                                           currentGlobalBestCost, flagsFromBranch0);
            if (cost0 < bestCostFound) {
                bestCostFound = cost0;
                currentBestFlags = flagsFromBranch0;
            }
        }
    }

    flags[index] = 1;
    {
        int cInc = calculateCost(G, flags, index, 1);
        if (currentCost + cInc < currentGlobalBestCost && currentCost + cInc < bestCostFound) {
             vector<int> flagsFromBranch1(V);
             int cost1 = resAlmostSeqSubTreeMPI(G, flags, index + 1, countA, currentCost + cInc,
                                            currentGlobalBestCost, flagsFromBranch1);
             if (cost1 < bestCostFound) {
                 bestCostFound = cost1;
                 currentBestFlags = flagsFromBranch1;
             }
        }
    }

    flags[index] = -1;

    if (bestCostFound != numeric_limits<int>::max()) {
        bestFlags_local = currentBestFlags;
    }

    return bestCostFound;
}

vector<BnBState> buildPartialStates(const Graph &G, int enoughStates, int maxDepth, const int& currentGlobalBestCost) {
    vector<BnBState> result;
    queue<BnBState> Q;

    BnBState init(V);
    init.index = 0;
    init.countA = 0;
    init.cost = 0;
    Q.push(init);

    while (!Q.empty()) {
        BnBState s = Q.front();
        Q.pop();

         if (s.cost >= currentGlobalBestCost) continue;
         if (s.countA > selA || (s.countA + (V - s.index)) < selA) continue;

        if (s.index >= maxDepth || s.index == V) {
            result.push_back(s);
            if (s.index == V) continue;
        }
        else {
            {
                BnBState s0 = s;
                s0.flags[s0.index] = 0;
                int cInc = calculateCost(G, s0.flags, s0.index, 0);
                s0.cost   += cInc;
                s0.countA += 1;
                s0.index  += 1;
                if (s0.cost < currentGlobalBestCost && s0.countA <= selA && (s0.countA + (V - s0.index)) >= selA) {
                    Q.push(s0);
                }
            }

            {
                BnBState s1 = s;
                s1.flags[s1.index] = 1;
                int cInc = calculateCost(G, s1.flags, s1.index, 1);
                s1.cost  += cInc;
                s1.index += 1;
                 if (s1.cost < currentGlobalBestCost && (s1.countA + (V - s1.index)) >= selA) {
                    Q.push(s1);
                }
            }
        }

        if ((int)result.size() >= enoughStates) break;
    }

     while(!Q.empty() && (int)result.size() < enoughStates) {
        BnBState s = Q.front(); Q.pop();
         if (s.cost < currentGlobalBestCost && s.countA <= selA && (s.countA + (V - s.index)) >= selA) {
            result.push_back(s);
         }
    }


    return result;
}


void broadcastGraph(Graph& g, int root, MPI_Comm comm) {
    vector<vector<int>> edgeCosts = g.getEdgeCostsMatrix();
    vector<int> flat_costs(V * V);
     for(int i = 0; i < V; ++i) {
        memcpy(flat_costs.data() + i * V, edgeCosts[i].data(), V * sizeof(int));
    }
    MPI_Bcast(flat_costs.data(), V * V, MPI_INT, root, comm);


    vector<int> adj_sizes(V);
    vector<pair<int, int>> flat_adj_list;
    if (root == [](){int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r;}()) {
        const auto& adjList = g.getAdjList();
        for(int i=0; i<V; ++i) {
            adj_sizes[i] = adjList[i].size();
            for(const auto& edge : adjList[i]) {
                flat_adj_list.push_back(edge);
            }
        }
    }
    MPI_Bcast(adj_sizes.data(), V, MPI_INT, root, comm);

    int total_edges = flat_adj_list.size();
    MPI_Bcast(&total_edges, 1, MPI_INT, root, comm);

    if (root != [](){int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r;}()) {
        flat_adj_list.resize(total_edges);
    }

    vector<int> flat_adj_ints(total_edges * 2);
     if (root == [](){int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r;}()) {
        for(int i=0; i<total_edges; ++i) {
            flat_adj_ints[2*i] = flat_adj_list[i].first;
            flat_adj_ints[2*i+1] = flat_adj_list[i].second;
        }
     }
    MPI_Bcast(flat_adj_ints.data(), total_edges * 2, MPI_INT, root, comm);

    if (root != [](){int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r;}()) {
        vector<vector<int>> receivedEdgeCosts(V, vector<int>(V));
        for(int i = 0; i < V; ++i) {
             memcpy(receivedEdgeCosts[i].data(), flat_costs.data() + i * V, V * sizeof(int));
        }
        g.setEdgeCosts(receivedEdgeCosts);

        vector<vector<pair<int,int>>> receivedAdjList(V);
        int current_pos = 0;
        for(int i=0; i<V; ++i) {
            for(int j=0; j<adj_sizes[i]; ++j) {
                int neighbor = flat_adj_ints[2*current_pos];
                int weight = flat_adj_ints[2*current_pos + 1];
                receivedAdjList[i].emplace_back(neighbor, weight);
                current_pos++;
            }
        }
        g.setAdjList(receivedAdjList);
    }
}


int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2 && rank == 0) {
        cerr << "Error: Requires at least 2 MPI processes." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Graph g(V);

    double startTime = 0.0, endTime = 0.0;
    int local_best_cost = numeric_limits<int>::max();

    if (rank == 0) {
        cout << "V=" << V << ", selA=" << selA << endl;
        cout << "Running with " << size << " processes (1 Master, " << size - 1 << " Slaves)." << endl;

        string filename = "data/graf_30_20.txt";
        if (argc > 1) filename = argv[1];
        cout << "Master loading graph from: " << filename << endl;
        loadFile(filename, g);
        cout << "Master: Graph loaded." << endl;

        g_bestFlags.resize(V, -1);

        startTime = MPI_Wtime();

        cout << "Master broadcasting graph data..." << endl;
        broadcastGraph(g, 0, MPI_COMM_WORLD);
        cout << "Master: Broadcast complete." << endl;

        cout << "Master generating initial states (BFS)..." << endl;
        int BFS_DEPTH = 8;
        vector<BnBState> workQueue = buildPartialStates(g, ENOUGH_STATES, BFS_DEPTH, g_bestCost);
        cout << "Master generated " << workQueue.size() << " initial work units." << endl;

        int workSent = 0;
        int workReceived = 0;
        int bestSlaveRank = -1;
        vector<bool> slave_busy(size, false);

        for (int slaveRank = 1; slaveRank < size; ++slaveRank) {
            if (!workQueue.empty()) {
                sendState(workQueue.back(), slaveRank, TAG_WORK_UNIT, MPI_COMM_WORLD);
                workQueue.pop_back();
                slave_busy[slaveRank] = true;
                workSent++;
            }
        }
         cout << "Master: Initial work sent (" << workSent << " units)." << endl;

        MPI_Status status;

        while (workReceived < workSent) {
            int slaveResultCost;
            MPI_Recv(&slaveResultCost, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            int sourceRank = status.MPI_SOURCE;
            workReceived++;
            slave_busy[sourceRank] = false;

            if (slaveResultCost < g_bestCost) {
                g_bestCost = slaveResultCost;
                bestSlaveRank = sourceRank;
                cout << "Master: New best cost = " << g_bestCost << " (from rank " << bestSlaveRank << ")" << endl;
                for (int i = 1; i < size; ++i) {
                    if (i != rank) {
                         MPI_Send(&g_bestCost, 1, MPI_INT, i, TAG_BEST_COST_UPDATE, MPI_COMM_WORLD);
                    }
                }
            }

            if (!workQueue.empty()) {
                sendState(workQueue.back(), sourceRank, TAG_WORK_UNIT, MPI_COMM_WORLD);
                workQueue.pop_back();
                slave_busy[sourceRank] = true;
                workSent++;
            }
        }

        cout << "Master: All generated work completed (" << workReceived << "/" << workSent << ")." << endl;

        if (bestSlaveRank != -1) {
            cout << "Master requesting final flags from rank " << bestSlaveRank << endl;
            int request_signal = 0;
            MPI_Send(&request_signal, 1, MPI_INT, bestSlaveRank, TAG_REQUEST_FLAGS, MPI_COMM_WORLD); // Send request FIRST
            g_bestFlags.resize(V);
            MPI_Recv(g_bestFlags.data(), V, MPI_INT, bestSlaveRank, TAG_FINAL_FLAGS, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Wait for flags
            cout << "Master received final flags." << endl;
        } else {
             cout << "Master: No better solution reported by slaves than initial bound." << endl;
        }

        cout << "Master sending termination signal to all slaves..." << endl;
        int terminate_signal = 0;
        for (int i = 1; i < size; ++i) {
            MPI_Send(&terminate_signal, 1, MPI_INT, i, TAG_TERMINATE, MPI_COMM_WORLD);
        }

        endTime = MPI_Wtime();

        cout << "\n--- Final Result ---" << endl;
        if (g_bestCost != numeric_limits<int>::max()) {
            cout << "Minimum cut cost: " << g_bestCost << endl;
             if (bestSlaveRank != -1){
                 cout << "Flags: ";
                 for (int i = 0; i < V; ++i) cout << g_bestFlags[i] << (i == V - 1 ? "" : " ");
                 cout << endl;
             } else {
                 cout << "(Flags not retrieved from slaves)" << endl;
             }
        } else {
             cout << "No feasible solution found." << endl;
        }
        cout << "Total time: " << (endTime - startTime) << " seconds" << endl;
        cout << "--------------------" << endl;

    }
    else {
        broadcastGraph(g, 0, MPI_COMM_WORLD);

        BnBState currentState(V);
        MPI_Status status;
        local_best_cost = numeric_limits<int>::max();
        int myBestOverallCost = numeric_limits<int>::max();
        vector<int> myBestOverallFlags(V, -1);

        while (true) {
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_TERMINATE) {
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
                break; // Exit loop
            }
            else if (status.MPI_TAG == TAG_BEST_COST_UPDATE) {
                MPI_Recv(&local_best_cost, 1, MPI_INT, 0, TAG_BEST_COST_UPDATE, MPI_COMM_WORLD, &status);
            }
            else if (status.MPI_TAG == TAG_WORK_UNIT) {
                recvState(currentState, 0, TAG_WORK_UNIT, MPI_COMM_WORLD, &status);

                vector<int> flagsFromSubtree(V);
                int resultCost = resAlmostSeqSubTreeMPI(g, currentState.flags, currentState.index,
                                                        currentState.countA, currentState.cost,
                                                        local_best_cost,
                                                        flagsFromSubtree);

                if (resultCost < myBestOverallCost) {
                    myBestOverallCost = resultCost;
                    myBestOverallFlags = flagsFromSubtree;
                }

                MPI_Send(&resultCost, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
            }
             else if (status.MPI_TAG == TAG_REQUEST_FLAGS) {
                 int dummy;
                 MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_REQUEST_FLAGS, MPI_COMM_WORLD, &status);
                 MPI_Send(myBestOverallFlags.data(), V, MPI_INT, 0, TAG_FINAL_FLAGS, MPI_COMM_WORLD);
             }
            else {
                 cerr << "Slave " << rank << " received unexpected tag: " << status.MPI_TAG << endl;
                 MPI_Abort(MPI_COMM_WORLD, 99);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}