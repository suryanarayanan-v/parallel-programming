#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <queue>
#include <sstream>
#include <limits>
#include <chrono>
#include <mpi.h>
#include <cstring>
#include <cassert>
#include <numeric>
#include <omp.h>

using namespace std;

int V = 30;
int selA = 15;
int ENOUGH_STATES = 10000;
const int BFS_DEPTH = 8;

const int TAG_GRAPH_ADJ_SIZE = 10;
const int TAG_GRAPH_ADJ_DATA = 11;
const int TAG_GRAPH_COSTS = 12;
const int TAG_WORK_UNIT = 13;
const int TAG_RESULT = 14;
const int TAG_TERMINATE = 15;
const int TAG_BEST_COST_UPDATE = 16;
const int TAG_REQUEST_FLAGS = 17;
const int TAG_FINAL_FLAGS = 18;

int g_bestCost = numeric_limits<int>::max();
vector<int> g_bestFlags;

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
        V_local = adjList.size();
    }

    void setEdgeCosts(const vector<vector<int>>& newEdgeCosts) {
        edgeCosts = newEdgeCosts;
        V_local = edgeCosts.size();
    }

    void addEdge(int u, int v, int w) {
         if (u < 0 || u >= V_local || v < 0 || v >= V_local) return;

        if (u >= adjList.size()) adjList.resize(V_local);
        if (v >= adjList.size()) adjList.resize(V_local);
        if (u >= edgeCosts.size() || v >= edgeCosts.size()) {
             edgeCosts.resize(V_local, vector<int>(V_local, 0));
             for(auto& row : edgeCosts) row.resize(V_local, 0);
        }


        adjList[u].emplace_back(v, w);
        edgeCosts[u][v] = w;
        edgeCosts[v][u] = w;
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
            if (i < adjList.size()) {
                for (auto &edge : adjList[i]) {
                    cout << " -> (" << edge.first << ", weight: " << edge.second << ")";
                }
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
    int rank_for_error_msg;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_for_error_msg);

    if (!infile) {
        cerr << "MPI Rank " << rank_for_error_msg << ": Error opening file: " << fileName << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string line;
    int expected_size = -1;
    bool header_read = false;

    if (getline(infile, line)) {
        stringstream ss_header(line);
        int size_val;
        if (ss_header >> size_val && ss_header.eof()) {
            expected_size = size_val;
            header_read = true;
        } else {
            infile.clear();
            infile.seekg(0, ios::beg);
        }
    } else {
         cerr << "MPI Rank " << rank_for_error_msg << ": Error: File '" << fileName << "' is empty." << endl;
         MPI_Abort(MPI_COMM_WORLD, 4);
    }

    vector<vector<int>> matrix;
    int lines_read_for_matrix = 0;

    while (getline(infile, line)) {
        stringstream ss_check(line);
        string segment;
        bool empty_line = true;
        while(ss_check >> segment) {
            empty_line = false;
            break;
        }
        if(empty_line) continue;

        stringstream ss(line);
        vector<int> row;
        int value;
        while (ss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
             matrix.push_back(row);
             lines_read_for_matrix++;
             if (header_read && lines_read_for_matrix == expected_size) {
                 break;
             }
        }
    }
    infile.close();

    int size_determined;
    if (header_read) {
        size_determined = expected_size;
        if (lines_read_for_matrix != expected_size) {
             cerr << "MPI Rank " << rank_for_error_msg << ": Error: Header specified size " << expected_size
                  << " but read " << lines_read_for_matrix << " matrix rows from file '" << fileName << "'." << endl;
             MPI_Abort(MPI_COMM_WORLD, 5);
        }
    } else {
        size_determined = lines_read_for_matrix;
    }

    V = size_determined;
    G = Graph(V);

    if (matrix.size() != (size_t)V) {
         cerr << "MPI Rank " << rank_for_error_msg
              << ": Internal Error: Matrix row count (" << matrix.size()
              << ") mismatch after determining V=" << V << " from file '" << fileName << "'." << endl;
        MPI_Abort(MPI_COMM_WORLD, 6);
    }

    for(int i = 0; i < V; ++i) {
        if (matrix[i].size() != (size_t)V) {
             cerr << "MPI Rank " << rank_for_error_msg
                  << ": Error: Row " << i << " in matrix from file '" << fileName
                  << "' has incorrect size (" << matrix[i].size() << "), expected V=" << V << "." << endl;
             if (header_read) cerr << "       (Header specified V=" << expected_size << ")" << endl;
             else cerr << "       (Determined V from row count)" << endl;
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        for(int j = i + 1; j < V; ++j) {
            if(matrix[i][j] != 0) {
                G.addEdge(i, j, matrix[i][j]);
            }
        }
    }
}


int calculateCost(const Graph &G, const vector<int> &flags, int node, int flag) {
    int cost = 0;
    int current_v = G.getVertexCount();


    for (int i = 0; i < node; ++i) {
         if (flags[i] != -1 && flags[i] != flag) {
            cost += G.getEdgeCost(i, node);
        }
    }
    return cost;
}


void sendStatePacked(const BnBState& state, int dest, int tag, MPI_Comm comm) {
    int v_size = state.flags.size();
    int buffer_size = 3 + v_size;
    vector<int> send_buffer(buffer_size);

    send_buffer[0] = state.index;
    send_buffer[1] = state.countA;
    send_buffer[2] = state.cost;
    if (v_size > 0) {
        memcpy(send_buffer.data() + 3, state.flags.data(), v_size * sizeof(int));
    }

    MPI_Send(send_buffer.data(), buffer_size, MPI_INT, dest, tag, comm);
}

void recvStatePacked(BnBState& state, int source, int tag, MPI_Comm comm, MPI_Status* status) {
    int v_size = state.flags.size();
    assert(v_size > 0 && "recvStatePacked: state.flags must be pre-sized.");
    int buffer_size = 3 + v_size;
    vector<int> recv_buffer(buffer_size);

    MPI_Recv(recv_buffer.data(), buffer_size, MPI_INT, source, tag, comm, status);

    state.index = recv_buffer[0];
    state.countA = recv_buffer[1];
    state.cost = recv_buffer[2];
    if (v_size > 0) {
        memcpy(state.flags.data(), recv_buffer.data() + 3, v_size * sizeof(int));
    }
}


void solveSubtreeRecursiveOMP(
    const Graph &G,
    vector<int> currentFlags,
    int index,
    int countA,
    int currentCost,
    const int& currentGlobalBestCostKnownToSlave,
    int& bestCostFoundInSubtree,
    vector<int>& bestFlags_local
    )
{
    if (currentCost >= currentGlobalBestCostKnownToSlave) {
        return;
    }
    if (countA > selA || (countA + (V - index)) < selA) {
        return;
    }

    if (index == V) {
        if (countA == selA) {
            #pragma omp critical
            {
                if (currentCost < bestCostFoundInSubtree) {
                    bestCostFoundInSubtree = currentCost;
                    bestFlags_local = currentFlags;
                }
            }
        }
        return;
    }


    int costIncrement0 = calculateCost(G, currentFlags, index, 0);
    if (currentCost + costIncrement0 < currentGlobalBestCostKnownToSlave && (countA + 1) <= selA) {
        vector<int> flags0 = currentFlags;
        flags0[index] = 0;

        #pragma omp task firstprivate(flags0, index, countA, currentCost, costIncrement0) \
                         shared(G, currentGlobalBestCostKnownToSlave, bestCostFoundInSubtree, bestFlags_local)
        {
             solveSubtreeRecursiveOMP(G, flags0, index + 1, countA + 1, currentCost + costIncrement0,
                                    currentGlobalBestCostKnownToSlave,
                                    bestCostFoundInSubtree, bestFlags_local);
        }
    }


    int costIncrement1 = calculateCost(G, currentFlags, index, 1);
    if (currentCost + costIncrement1 < currentGlobalBestCostKnownToSlave &&
        currentCost + costIncrement1 < bestCostFoundInSubtree)
    {
        vector<int> flags1 = currentFlags;
        flags1[index] = 1;

        #pragma omp task firstprivate(flags1, index, countA, currentCost, costIncrement1) \
                         shared(G, currentGlobalBestCostKnownToSlave, bestCostFoundInSubtree, bestFlags_local)
        {
            solveSubtreeRecursiveOMP(G, flags1, index + 1, countA, currentCost + costIncrement1,
                                     currentGlobalBestCostKnownToSlave,
                                     bestCostFoundInSubtree, bestFlags_local);
        }
    }

}


vector<BnBState> buildPartialStates(const Graph &G, int enoughStatesTarget, int maxDepth, const int& currentGlobalBestCost) {
    vector<BnBState> result;
    queue<BnBState> Q;
    int currentV = G.getVertexCount();

    BnBState init(currentV);
    init.index = 0;
    init.countA = 0;
    init.cost = 0;
    Q.push(init);

    while (!Q.empty() && (int)result.size() < enoughStatesTarget) {
        BnBState s = Q.front();
        Q.pop();

         if (s.cost >= currentGlobalBestCost) continue;
         if (s.countA > selA || (s.countA + (currentV - s.index)) < selA) continue;

        if (s.index >= maxDepth || s.index == currentV) {
             if (s.cost < currentGlobalBestCost && s.countA <= selA && (s.countA + (currentV - s.index)) >= selA) {
                 result.push_back(s);
             }
            if (s.index == currentV || s.index >= maxDepth) continue;
        }
        else {
            BnBState s0 = s;
            s0.flags[s0.index] = 0;
            int cInc0 = calculateCost(G, s0.flags, s0.index, 0);
            s0.cost   += cInc0;
            s0.countA += 1;
            s0.index  += 1;
            if (s0.cost < currentGlobalBestCost && s0.countA <= selA && (s0.countA + (currentV - s0.index)) >= selA) {
                Q.push(s0);
            }

            BnBState s1 = s;
            s1.flags[s1.index] = 1;
            int cInc1 = calculateCost(G, s1.flags, s1.index, 1);
            s1.cost  += cInc1;
            s1.index += 1;
             if (s1.cost < currentGlobalBestCost && (s1.countA + (currentV - s1.index)) >= selA) {
                Q.push(s1);
            }
        }
    }

    while(!Q.empty() && (int)result.size() < enoughStatesTarget) {
        BnBState s = Q.front(); Q.pop();
         if (s.cost < currentGlobalBestCost && s.countA <= selA && (s.countA + (currentV - s.index)) >= selA) {
            result.push_back(s);
         }
    }
    return result;
}

void broadcastGraph(Graph& g, int root, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    int currentV_bcast = 0;

    if (rank == root) {
        currentV_bcast = g.getVertexCount();
    }
    MPI_Bcast(&currentV_bcast, 1, MPI_INT, root, comm);

    V = currentV_bcast;
    if (rank != root) {
        g = Graph(V);
    } else {
        if (g.getVertexCount() != V) {
             g = Graph(V);
        }
    }
    g_bestFlags.resize(V);


    vector<int> flat_costs(V * V);
    if (rank == root) {
        const auto& edgeCosts = g.getEdgeCostsMatrix();
        assert((int)edgeCosts.size() == V);
        for(int i = 0; i < V; ++i) {
            assert((int)edgeCosts[i].size() == V);
            memcpy(flat_costs.data() + i * V, edgeCosts[i].data(), V * sizeof(int));
        }
    }
    MPI_Bcast(flat_costs.data(), V * V, MPI_INT, root, comm);
    if (rank != root) {
        vector<vector<int>> receivedEdgeCosts(V, vector<int>(V));
        for(int i = 0; i < V; ++i) {
             memcpy(receivedEdgeCosts[i].data(), flat_costs.data() + i * V, V * sizeof(int));
        }
        g.setEdgeCosts(receivedEdgeCosts);
    }


    vector<int> adj_sizes(V);
    vector<pair<int, int>> flat_adj_list;
    int total_edges = 0;

    if (rank == root) {
        const auto& adjList = g.getAdjList();
        assert((int)adjList.size() == V);
        flat_adj_list.clear();
        for(int i=0; i<V; ++i) {
            adj_sizes[i] = adjList[i].size();
            for(const auto& edge : adjList[i]) {
                flat_adj_list.push_back(edge);
            }
        }
        total_edges = flat_adj_list.size();
    }

    MPI_Bcast(adj_sizes.data(), V, MPI_INT, root, comm);
    MPI_Bcast(&total_edges, 1, MPI_INT, root, comm);

    vector<int> flat_adj_ints(total_edges * 2);
    if (rank == root) {
        for(int i=0; i<total_edges; ++i) {
            flat_adj_ints[2*i]     = flat_adj_list[i].first;
            flat_adj_ints[2*i + 1] = flat_adj_list[i].second;
        }
    }
    MPI_Bcast(flat_adj_ints.data(), total_edges * 2, MPI_INT, root, comm);

    if (rank != root) {
        vector<vector<pair<int,int>>> receivedAdjList(V);
        int current_pos = 0;
        for(int i=0; i<V; ++i) {
            receivedAdjList[i].reserve(adj_sizes[i]);
            for(int j=0; j<adj_sizes[i]; ++j) {
                if (2*current_pos + 1 >= flat_adj_ints.size()) {
                    cerr << "Rank " << rank << ": Error reconstructing adj list, index out of bounds." << endl;
                    MPI_Abort(MPI_COMM_WORLD, 7);
                }
                int neighbor = flat_adj_ints[2*current_pos];
                int weight   = flat_adj_ints[2*current_pos + 1];
                receivedAdjList[i].emplace_back(neighbor, weight);
                current_pos++;
            }
        }
         if (current_pos != total_edges) {
             cerr << "Rank " << rank << ": Error reconstructing adj list, edge count mismatch. Expected "
                  << total_edges << ", processed " << current_pos << endl;
             MPI_Abort(MPI_COMM_WORLD, 8);
         }
        g.setAdjList(receivedAdjList);
    }
}


int main(int argc, char *argv[]) {
    int rank, size;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_support);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0 && size > 1) {
        ENOUGH_STATES = max(200 * (size - 1), 1000);
    }

    Graph g;
    double startTime = 0.0, endTime = 0.0;
    int slave_local_best_cost = numeric_limits<int>::max();
    vector<int> slave_local_best_flags;

    if (rank == 0) {
        cout << "MPI+OpenMP Branch and Bound Solver" << endl;
        cout << "----------------------------------" << endl;
        cout << "V=" << V << ", selA=" << selA << " (Initial values, V might change based on file)" << endl;
        cout << "Running with " << size << " MPI processes." << endl;
        #pragma omp parallel
        {
            #pragma omp master
            {
                 cout << "Master (Rank 0) is using " << omp_get_num_threads() << " OpenMP threads." << endl;
            }
        }
        cout << "Target work units: " << ENOUGH_STATES << ", Initial BFS Depth: " << BFS_DEPTH << endl;

        string filename = "data/graf_30_20.txt";
        if (argc > 1) filename = argv[1];
        cout << "Master loading graph from: " << filename << endl;

        loadFile(filename, g);
        V = g.getVertexCount();
        cout << "Master: Graph loaded with V=" << V << "." << endl;
        selA = V / 2;
        cout << "Master: Updated selA=" << selA << " based on actual V=" << V << "." << endl;


        g_bestFlags.assign(V, -1);

        startTime = MPI_Wtime();

        cout << "Master broadcasting graph data (V=" << V << ")..." << endl;
        broadcastGraph(g, 0, MPI_COMM_WORLD);
        cout << "Master: Broadcast complete." << endl;

        if (size == 1) {
             cout << "Master: Only 1 process, running sequentially with OpenMP..." << endl;
             BnBState initial_state(V);
             slave_local_best_flags.resize(V);

             int final_best_cost = numeric_limits<int>::max();
             vector<int> final_best_flags(V, -1);

             #pragma omp parallel shared(g, final_best_cost, final_best_flags)
             {
                 #pragma omp single
                 {
                      solveSubtreeRecursiveOMP(g, initial_state.flags, initial_state.index,
                                            initial_state.countA, initial_state.cost,
                                            g_bestCost,
                                            final_best_cost, final_best_flags);
                 }
             }

             g_bestCost = final_best_cost;
             g_bestFlags = final_best_flags;
             cout << "Master: Sequential OpenMP run finished." << endl;
        }
        else {
            cout << "Master generating initial states (BFS)..." << endl;
            vector<BnBState> workQueue = buildPartialStates(g, ENOUGH_STATES, BFS_DEPTH, g_bestCost);
            cout << "Master generated " << workQueue.size() << " initial work units." << endl;

            if (workQueue.empty() && V > 0) {
                 cout << "Warning: No initial work units generated. Running root sequentially..." << endl;
                 BnBState initial_state(V);
                 slave_local_best_flags.resize(V);
                 int final_best_cost = numeric_limits<int>::max();
                 vector<int> final_best_flags(V,-1);
                 #pragma omp parallel shared(g, final_best_cost, final_best_flags)
                 {
                     #pragma omp single
                     {
                          solveSubtreeRecursiveOMP(g, initial_state.flags, initial_state.index,
                                                initial_state.countA, initial_state.cost,
                                                g_bestCost,
                                                final_best_cost, final_best_flags);
                     }
                 }
                 g_bestCost = final_best_cost;
                 g_bestFlags = final_best_flags;
                 cout << "Master: Fallback sequential run finished." << endl;
            }

            int workSent = 0;
            int workReceived = 0;
            int bestSlaveRank = -1;
            vector<bool> slave_busy(size, false);
            vector<MPI_Request> cost_update_requests;

            for (int slaveRank = 1; slaveRank < size; ++slaveRank) {
                if (!workQueue.empty()) {
                    workQueue.back().flags.resize(V);
                    sendStatePacked(workQueue.back(), slaveRank, TAG_WORK_UNIT, MPI_COMM_WORLD);
                    workQueue.pop_back();
                    slave_busy[slaveRank] = true;
                    workSent++;
                } else {
                     slave_busy[slaveRank] = false;
                }
            }
            cout << "Master: Initial work sent (" << workSent << " units to " << min(workSent, size-1) << " slaves)." << endl;

            MPI_Status status;
            int active_slaves = workSent;

            while (active_slaves > 0) {
                int slaveResultCost;
                MPI_Recv(&slaveResultCost, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
                int sourceRank = status.MPI_SOURCE;

                workReceived++;
                bool new_best_found = false;

                if (slaveResultCost < g_bestCost) {
                    g_bestCost = slaveResultCost;
                    bestSlaveRank = sourceRank;
                    new_best_found = true;
                    cout << "Master: New best cost = " << g_bestCost << " (reported by rank " << bestSlaveRank << ")" << endl;

                    for (int i = 1; i < size; ++i) {
                        MPI_Request req;
                        MPI_Isend(&g_bestCost, 1, MPI_INT, i, TAG_BEST_COST_UPDATE, MPI_COMM_WORLD, &req);
                        cost_update_requests.push_back(req);
                    }
                }

                if (!workQueue.empty()) {
                    workQueue.back().flags.resize(V);
                    sendStatePacked(workQueue.back(), sourceRank, TAG_WORK_UNIT, MPI_COMM_WORLD);
                    workQueue.pop_back();
                } else {
                    slave_busy[sourceRank] = false;
                    active_slaves--;
                }

                if (!cost_update_requests.empty()) {
                    int completed_count = 0;
                    vector<MPI_Status> statuses(cost_update_requests.size());
                    MPI_Testall(cost_update_requests.size(), cost_update_requests.data(), &completed_count, statuses.data());
                    cost_update_requests.clear();
                }
            }

            cout << "Master: All generated work completed (" << workReceived << "/" << workSent << ")." << endl;

            if (!cost_update_requests.empty()) {
                MPI_Waitall(cost_update_requests.size(), cost_update_requests.data(), MPI_STATUSES_IGNORE);
                cost_update_requests.clear();
            }

            if (bestSlaveRank != -1 && g_bestCost != numeric_limits<int>::max()) {
                cout << "Master requesting final flags from rank " << bestSlaveRank << endl;
                int request_signal = 0;
                MPI_Send(&request_signal, 1, MPI_INT, bestSlaveRank, TAG_REQUEST_FLAGS, MPI_COMM_WORLD);
                g_bestFlags.resize(V);
                MPI_Recv(g_bestFlags.data(), V, MPI_INT, bestSlaveRank, TAG_FINAL_FLAGS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cout << "Master received final flags." << endl;
            } else if (g_bestCost == numeric_limits<int>::max()) {
                 cout << "Master: No feasible solution found by slaves (or initial sequential run)." << endl;
            } else {
                 cout << "Master: Best cost " << g_bestCost << " likely came from initial bound or sequential phase." << endl;
            }


            cout << "Master sending termination signal to all slaves..." << endl;
            int terminate_signal = 0;
            vector<MPI_Request> terminate_requests;
            for (int i = 1; i < size; ++i) {
                 MPI_Request req;
                 MPI_Isend(&terminate_signal, 1, MPI_INT, i, TAG_TERMINATE, MPI_COMM_WORLD, &req);
                 terminate_requests.push_back(req);
            }
            if (!terminate_requests.empty()) {
                 MPI_Waitall(terminate_requests.size(), terminate_requests.data(), MPI_STATUSES_IGNORE);
            }
            cout << "Master: Termination signals sent." << endl;
        }

        endTime = MPI_Wtime();

        cout << "\n--- Final Result (Master Rank " << rank << ") ---" << endl;
        if (g_bestCost != numeric_limits<int>::max()) {
            cout << "Minimum cut cost: " << g_bestCost << endl;

            bool flags_valid = (g_bestFlags.size() == V);
            for(int flag_val : g_bestFlags) { if (flag_val == -1) flags_valid = false; break;}

            if (flags_valid){
                 cout << "Flags: ";
                 int count0 = 0;
                 for (int i = 0; i < V; ++i) {
                    cout << g_bestFlags[i] << (i == V - 1 ? "" : " ");
                    if(g_bestFlags[i] == 0) count0++;
                 }
                 cout << endl;
                 cout << "(Nodes in group 0: " << count0 << ", group 1: " << V-count0 << ", Target selA: " << selA << ")" << endl;
            } else {
                 cout << "(Final flags not available or potentially incomplete)" << endl;
            }
        } else {
             cout << "No feasible solution found." << endl;
        }
        cout << "Total time: " << (endTime - startTime) << " seconds" << endl;
        cout << "---------------------------------" << endl;

    }
    else {
        broadcastGraph(g, 0, MPI_COMM_WORLD);
        V = g.getVertexCount();
        selA = V / 2;
        slave_local_best_flags.resize(V);

        if (rank == 1) {
           #pragma omp parallel
           {
               #pragma omp master
               {
                  cout << "Slave (Rank " << rank << ") is using " << omp_get_num_threads() << " OpenMP threads." << endl;
               }
           }
        }

        BnBState currentState(V);
        MPI_Status status;
        int currentGlobalBestCostKnownToSlave = numeric_limits<int>::max();
        int myBestResultCost = numeric_limits<int>::max();
        vector<int> myBestResultFlags(V, -1);

        while (true) {
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int message_tag = status.MPI_TAG;

            if (message_tag == TAG_TERMINATE) {
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_TERMINATE, MPI_COMM_WORLD, &status);
                break;
            }
            else if (message_tag == TAG_BEST_COST_UPDATE) {
                MPI_Recv(&currentGlobalBestCostKnownToSlave, 1, MPI_INT, 0, TAG_BEST_COST_UPDATE, MPI_COMM_WORLD, &status);
            }
            else if (message_tag == TAG_WORK_UNIT) {
                currentState.flags.resize(V);
                recvStatePacked(currentState, 0, TAG_WORK_UNIT, MPI_COMM_WORLD, &status);

                int resultCostFromSubtree = numeric_limits<int>::max();
                vector<int> flagsFromSubtree(V,-1);

                #pragma omp parallel shared(g, currentGlobalBestCostKnownToSlave, resultCostFromSubtree, flagsFromSubtree) \
                                 firstprivate(currentState)
                {
                    #pragma omp single
                    {
                        solveSubtreeRecursiveOMP(g, currentState.flags, currentState.index,
                                                 currentState.countA, currentState.cost,
                                                 currentGlobalBestCostKnownToSlave,
                                                 resultCostFromSubtree,
                                                 flagsFromSubtree);
                    }
                }

                if (resultCostFromSubtree < myBestResultCost) {
                    myBestResultCost = resultCostFromSubtree;
                    myBestResultFlags = flagsFromSubtree;
                }

                MPI_Send(&resultCostFromSubtree, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
            }
             else if (message_tag == TAG_REQUEST_FLAGS) {
                 int dummy;
                 MPI_Recv(&dummy, 1, MPI_INT, 0, TAG_REQUEST_FLAGS, MPI_COMM_WORLD, &status);

                 if (myBestResultFlags.size() != V) {
                     cerr << "Slave " << rank << " Error: myBestResultFlags has incorrect size (" << myBestResultFlags.size() << " vs V=" << V << ")" << endl;
                     myBestResultFlags.assign(V, -2);
                 } else if (myBestResultCost == numeric_limits<int>::max()) {
                      myBestResultFlags.assign(V, -1);
                 }
                 MPI_Send(myBestResultFlags.data(), V, MPI_INT, 0, TAG_FINAL_FLAGS, MPI_COMM_WORLD);
             }
            else {
                 int unexpected_tag = message_tag;
                 char error_buf[MPI_MAX_ERROR_STRING];
                 int len;
                 MPI_Error_string(status.MPI_ERROR, error_buf, &len);
                 cerr << "Slave " << rank << " received unexpected tag: " << unexpected_tag
                      << " from source " << status.MPI_SOURCE << ". Error: " << error_buf << endl;
                 int dummy_recv;
                 MPI_Recv(&dummy_recv, 1, MPI_INT, status.MPI_SOURCE, unexpected_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}