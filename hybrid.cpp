#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <queue>
#include <sstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <tuple> // Added for std::tuple

using namespace std;

int V = 30;
int selA = 15;
int ENOUGH_STATES = 10000;
int BFS_DEPTH = 8;

class Graph {
    int V;
    vector<vector<pair<int, int>>> adjList;
    vector<vector<int>> edgeCosts;

public:
    Graph() : V(0) {}

    explicit Graph(int V) : V(V) {
        adjList.resize(V);
        edgeCosts.resize(V, vector<int>(V, 0));
    }

    void initialize(int num_vertices) {
         V = num_vertices;
         adjList.assign(V, vector<pair<int,int>>());
         edgeCosts.assign(V, vector<int>(V, 0));
    }

    void addEdge(int u, int v, int w) {
        if (u < 0 || u >= V || v < 0 || v >= V) {
             cerr << "Warning: Attempting to add edge with invalid vertex index (" << u << ", " << v << "). Max V=" << V << endl;
             return;
        }
        adjList[u].emplace_back(v, w);
        adjList[v].emplace_back(u, w);
        edgeCosts[u][v] = w;
        edgeCosts[v][u] = w;
    }

    const vector<pair<int, int>>& getEdges(int u) const {
         if (u < 0 || u >= V) throw std::out_of_range("Vertex index out of range in getEdges");
         return adjList[u];
    }

    int getEdgeCost(int u, int v) const {
        if (u < 0 || u >= V || v < 0 || v >= V) {
             return 0;
        }
        return edgeCosts[u][v];
    }

    int getVertexCount() const {
        return V;
    }

    void printGraph(int rank = -1) {
        cout << "--- Graph (Rank " << rank << ", V=" << V << ") ---" << endl;
        for (int i = 0; i < V; ++i) {
            cout << "Vertex " << i << ":";
             for (const auto &edge : getEdges(i)) {
                 cout << " -> (" << edge.first << ", weight: " << edge.second << ")";
             }
             cout << endl;
        }
         cout << "--- End Graph ---" << endl;
    }
};

struct BnBState {
    vector<int> flags;
    int index;
    int countA;
    int cost;
};

void serializeGraph(const Graph& G, vector<int>& buffer) {
    buffer.clear();
    int V_count = G.getVertexCount();
    buffer.push_back(V_count);

    vector<tuple<int, int, int>> edges;
    for (int u = 0; u < V_count; ++u) {
        for (const auto& edge : G.getEdges(u)) {
            int v = edge.first;
            int w = edge.second;
            if (u < v) {
                edges.emplace_back(u, v, w);
            }
        }
    }

    buffer.push_back(edges.size());

    for (const auto& edge_tuple : edges) {
        buffer.push_back(get<0>(edge_tuple));
        buffer.push_back(get<1>(edge_tuple));
        buffer.push_back(get<2>(edge_tuple));
    }
}

void deserializeGraph(Graph& G, const vector<int>& buffer) {
    if (buffer.size() < 2) {
        throw runtime_error("Graph buffer too small for header.");
    }
    int V_count = buffer[0];
    int num_edges = buffer[1];

    G.initialize(V_count);

    if (buffer.size() < (size_t)(2 + num_edges * 3)) {
         cerr << "Warning: Graph buffer size mismatch. Expected at least " << (2 + num_edges * 3) << " elements, got " << buffer.size() << "." << endl;
         if (buffer.size() < 2) return;
    }


    int buffer_idx = 2;
    for (int i = 0; i < num_edges; ++i) {
        if (buffer_idx + 2 >= buffer.size()) {
             cerr << "Error: Ran out of bounds in graph buffer during deserialization at edge " << i << "." << endl;
             break;
        }
        int u = buffer[buffer_idx++];
        int v = buffer[buffer_idx++];
        int w = buffer[buffer_idx++];
        G.addEdge(u, v, w);
    }
}

void serializeState(const BnBState& state, vector<int>& buffer, int V_count) {
    buffer.clear();
    buffer.reserve(3 + V_count);
    buffer.push_back(state.index);
    buffer.push_back(state.countA);
    buffer.push_back(state.cost);
    if (state.flags.size() != (size_t)V_count) {
         throw runtime_error("State flags vector size mismatch during serialization.");
    }
    buffer.insert(buffer.end(), state.flags.begin(), state.flags.end());
}

BnBState deserializeState(const vector<int>& buffer, int start_idx, int V_count) {
    BnBState state;
    int expected_size = 3 + V_count;
     if (start_idx + expected_size > buffer.size()) {
         throw runtime_error("Buffer out of bounds during state deserialization.");
     }

    state.index = buffer[start_idx];
    state.countA = buffer[start_idx + 1];
    state.cost = buffer[start_idx + 2];
    state.flags.assign(buffer.begin() + start_idx + 3, buffer.begin() + start_idx + 3 + V_count);
     if (state.flags.size() != (size_t)V_count) {
         throw runtime_error("Deserialized state flags vector size mismatch.");
     }
    return state;
}

void loadFile(const string& fileName, Graph &G) {
    ifstream infile(fileName);
    if (!infile) {
        cerr << "Error opening file: " << fileName << endl;
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

    if (matrix.empty()) {
         cerr << "Error: No data read from file or file format incorrect." << endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int size = (int)matrix.size();
     if (size != G.getVertexCount()){
          cerr << "Warning: Matrix size (" << size << ") in file differs from graph V (" << G.getVertexCount() << "). Using matrix size." << endl;
          V = size;
          G.initialize(V);
     }


    for(int i=0; i<size; ++i) {
        if (matrix[i].size() != (size_t)size) {
             cerr << "Error: Matrix row " << i << " size mismatch. Expected " << size << " columns, got " << matrix[i].size() << "." << endl;
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for(int j=0; j<size; ++j) {
            if(matrix[i][j] != 0 && i < j) {
                G.addEdge(i, j, matrix[i][j]);
            }
        }
    }
}

int calculateCost(const Graph &G, const vector<int> &flags, int node, int flag) {
    int cost = 0;
    int V_count = G.getVertexCount();

    try {
        const auto &neighbors = G.getEdges(node);
        for (const auto &nbr : neighbors) {
            int nnode = nbr.first;
            int weight = nbr.second;
            if (nnode >= 0 && nnode < V_count && flags[nnode] != -1 && flags[nnode] != flag) {
                cost += weight;
            }
        }
    } catch (const std::out_of_range& oor) {
         cerr << "Error accessing neighbors for node " << node << ": " << oor.what() << endl;
         return numeric_limits<int>::max() / 2;
    }

    return cost;
}

void resAlmostSeqSubTree(const Graph &G, vector<int> &flags, int index, int countA, int currentCost,
                         int& local_bestCost, vector<int>& local_bestFlags,
                         const int V_count, const int targetA)
{
    if (currentCost >= local_bestCost) return;

    if (countA > targetA || (countA + (V_count - index)) < targetA) return;

    if (index == V_count) {
        if (countA == targetA) {
            #pragma omp critical
            {
                if (currentCost < local_bestCost) {
                    local_bestCost  = currentCost;
                    local_bestFlags = flags;
                }
            }
        }
        return;
    }

    flags[index] = 0;
    int cInc0 = calculateCost(G, flags, index, 0);
     if (currentCost <= numeric_limits<int>::max() - cInc0) {
        resAlmostSeqSubTree(G, flags, index + 1, countA + 1, currentCost + cInc0,
                            local_bestCost, local_bestFlags, V_count, targetA);
     }

    flags[index] = 1;
    int cInc1 = calculateCost(G, flags, index, 1);
     if (currentCost <= numeric_limits<int>::max() - cInc1) {
         resAlmostSeqSubTree(G, flags, index + 1, countA, currentCost + cInc1,
                             local_bestCost, local_bestFlags, V_count, targetA);
     }

    flags[index] = -1;
}

vector<BnBState> buildPartialStates(const Graph &G, int enoughStates, int maxDepth, int& current_best_cost_estimate) {
    vector<BnBState> result;
    queue<BnBState> Q;
    int V_count = G.getVertexCount();

    BnBState init;
    init.flags.assign(V_count, -1);
    init.index  = 0;
    init.countA = 0;
    init.cost   = 0;
    Q.push(init);

    while (!Q.empty() && (int)result.size() < enoughStates) {
        BnBState s = Q.front();
        Q.pop();

        if (s.cost >= current_best_cost_estimate) continue;

        if (s.index >= maxDepth || s.index == V_count) {
             if (s.flags.size() == (size_t)V_count) {
                 result.push_back(s);
             } else {
                  cerr << "Warning: BFS generated state with incorrect flag size (" << s.flags.size() << " vs V=" << V_count << "). Skipping." << endl;
             }
             if ((int)result.size() >= enoughStates) break;
        }
        else {
            {
                BnBState s0 = s;
                s0.flags[s0.index] = 0;
                int cInc = calculateCost(G, s0.flags, s0.index, 0);

                if (s0.cost <= numeric_limits<int>::max() - cInc) {
                    s0.cost += cInc;
                    s0.countA += 1;
                    s0.index += 1;

                    if (s0.cost < current_best_cost_estimate && s0.countA <= selA && (s0.countA + (V_count - s0.index)) >= selA)
                    {
                        Q.push(s0);
                    }
                }
            }

            {
                 BnBState s1 = s;
                 s1.flags[s1.index] = 1;
                 int cInc = calculateCost(G, s1.flags, s1.index, 1);

                 if (s1.cost <= numeric_limits<int>::max() - cInc) {
                      s1.cost += cInc;
                      s1.index += 1;

                      if (s1.cost < current_best_cost_estimate && (s1.countA + (V_count - s1.index)) >= selA)
                      {
                          Q.push(s1);
                      }
                 }
            }
        }
    }

    while (!Q.empty() && (int)result.size() < enoughStates) {
         BnBState s = Q.front();
         Q.pop();
         if (s.index >= maxDepth || s.index == V_count) {
             if (s.flags.size() == (size_t)V_count) {
                 result.push_back(s);
             } else {
                  cerr << "Warning: BFS (drain) generated state with incorrect flag size (" << s.flags.size() << " vs V=" << V_count << "). Skipping." << endl;
             }
         } else {
              if (s.flags.size() == (size_t)V_count) {
                 result.push_back(s);
              }
         }
    }

    return result;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    Graph g;
    vector<BnBState> localPartialStates;
    vector<int> local_bestFlags;
    int local_bestCost = numeric_limits<int>::max();
    double start_time, end_time, local_start_time, local_end_time;

    if (world_rank == 0) {
        cout << "MPI World Size: " << world_size << endl;
        cout << "OpenMP Max Threads (per process): " << omp_get_max_threads() << endl;

        Graph temp_g(V);
        loadFile("data/graf_30_20.txt", temp_g);
        V = temp_g.getVertexCount();
        g = temp_g;


        start_time = MPI_Wtime();

        MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&selA, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ENOUGH_STATES, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&BFS_DEPTH, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> graph_buffer;
        serializeGraph(g, graph_buffer);
        int graph_buffer_size = graph_buffer.size();
        MPI_Bcast(&graph_buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(graph_buffer.data(), graph_buffer_size, MPI_INT, 0, MPI_COMM_WORLD);

        int bfs_prune_cost = numeric_limits<int>::max();
        vector<BnBState> allPartialStates = buildPartialStates(g, ENOUGH_STATES, BFS_DEPTH, bfs_prune_cost);
        cout << "Rank 0: Generated " << allPartialStates.size() << " partial states from BFS (Depth " << BFS_DEPTH << ")." << endl;

        if (allPartialStates.empty() && V > 0) {
            cout << "Rank 0: Warning - BFS generated no partial states. Problem might be too small or BFS depth too large?" << endl;
            BnBState init;
            init.flags.assign(V, -1);
            init.index = 0;
            init.countA = 0;
            init.cost = 0;
            allPartialStates.push_back(init);
            cout << "Rank 0: Added root state as the only partial state." << endl;
        }

        vector<int> all_states_buffer;
        vector<int> state_sizes;
        int state_elem_size = 3 + V;

        for (const auto& state : allPartialStates) {
             if (state.flags.size() != (size_t)V) {
                  cerr << "Rank 0 Error: State found with incorrect flag size (" << state.flags.size() << ") before serialization. Aborting." << endl;
                  MPI_Abort(MPI_COMM_WORLD, 2);
             }
            vector<int> single_state_buffer;
            serializeState(state, single_state_buffer, V);
            if (single_state_buffer.size() != (size_t)state_elem_size) {
                 cerr << "Rank 0 Error: Serialized state size mismatch. Expected " << state_elem_size << ", got " << single_state_buffer.size() << ". Aborting." << endl;
                 MPI_Abort(MPI_COMM_WORLD, 3);
            }
            state_sizes.push_back(single_state_buffer.size());
            all_states_buffer.insert(all_states_buffer.end(), single_state_buffer.begin(), single_state_buffer.end());
        }

        int num_total_states = allPartialStates.size();
        vector<int> states_per_process(world_size);
        vector<int> sendcounts(world_size);
        vector<int> displs(world_size);

        int base_states = num_total_states / world_size;
        int extra_states = num_total_states % world_size;
        int current_displ = 0;
        int states_assigned = 0;

        for (int i = 0; i < world_size; ++i) {
            states_per_process[i] = base_states + (i < extra_states ? 1 : 0);
            sendcounts[i] = states_per_process[i] * state_elem_size;
            displs[i] = current_displ;
            current_displ += sendcounts[i];
            states_assigned += states_per_process[i];
        }
         if(states_assigned != num_total_states) {
             cerr << "Rank 0 Error: State assignment mismatch (" << states_assigned << " vs " << num_total_states << "). Aborting." << endl;
             MPI_Abort(MPI_COMM_WORLD, 4);
         }
          if(current_displ != all_states_buffer.size()) {
              cerr << "Rank 0 Error: Total buffer displacement mismatch (" << current_displ << " vs " << all_states_buffer.size() << "). Aborting." << endl;
              MPI_Abort(MPI_COMM_WORLD, 5);
          }

        int local_num_states;
        MPI_Scatter(states_per_process.data(), 1, MPI_INT,
                    &local_num_states, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> local_state_buffer(sendcounts[world_rank]);
        MPI_Scatterv(all_states_buffer.data(), sendcounts.data(), displs.data(), MPI_INT,
                     local_state_buffer.data(), sendcounts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

        localPartialStates.reserve(local_num_states);
        for (int i = 0; i < local_num_states; ++i) {
            localPartialStates.push_back(deserializeState(local_state_buffer, i * state_elem_size, V));
        }
         cout << "Rank 0: Received and deserialized " << localPartialStates.size() << " states." << endl;

    } else {

        MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&selA, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ENOUGH_STATES, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&BFS_DEPTH, 1, MPI_INT, 0, MPI_COMM_WORLD);

        g.initialize(V);

        int graph_buffer_size;
        MPI_Bcast(&graph_buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        vector<int> graph_buffer(graph_buffer_size);
        MPI_Bcast(graph_buffer.data(), graph_buffer_size, MPI_INT, 0, MPI_COMM_WORLD);
        deserializeGraph(g, graph_buffer);

        int local_num_states;
        MPI_Scatter(nullptr, 0, MPI_INT,
                    &local_num_states, 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> sendcounts(world_size);
        vector<int> displs(world_size);
        int state_elem_size = 3 + V;
        int local_buffer_size = local_num_states * state_elem_size;
        vector<int> local_state_buffer(local_buffer_size);

        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(), MPI_INT,
                     local_state_buffer.data(), local_buffer_size, MPI_INT, 0, MPI_COMM_WORLD);

        localPartialStates.reserve(local_num_states);
        for (int i = 0; i < local_num_states; ++i) {
            try {
                 localPartialStates.push_back(deserializeState(local_state_buffer, i * state_elem_size, V));
            } catch (const std::runtime_error& e) {
                 cerr << "Rank " << world_rank << " Error deserializing state " << i << ": " << e.what() << ". Buffer size=" << local_state_buffer.size() << ", index=" << i * state_elem_size << ", V=" << V << ". Aborting." << endl;
                 MPI_Abort(MPI_COMM_WORLD, 6);
            }
        }
    }

    local_bestFlags.assign(V, -1);
    local_bestCost = numeric_limits<int>::max();

    MPI_Barrier(MPI_COMM_WORLD);
    local_start_time = MPI_Wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)localPartialStates.size(); i++) {
        BnBState st = localPartialStates[i];

        vector<int> thread_localFlags = st.flags;

         if (thread_localFlags.size() == (size_t)V) {
             resAlmostSeqSubTree(g, thread_localFlags, st.index, st.countA, st.cost,
                                 local_bestCost, local_bestFlags, V, selA);
         } else {
              #pragma omp critical
              {
                  cerr << "Rank " << world_rank << ", Thread " << omp_get_thread_num()
                       << ": Error - Incorrect flag size (" << thread_localFlags.size() << " vs V=" << V
                       << ") for state " << i << ". Skipping." << endl;
              }
         }
    }

    local_end_time = MPI_Wtime();

    struct {
        int cost;
        int rank;
    } local_min_info, global_min_info;

    local_min_info.cost = local_bestCost;
    local_min_info.rank = world_rank;

    MPI_Allreduce(&local_min_info, &global_min_info, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        end_time = MPI_Wtime();

        if (global_min_info.cost == numeric_limits<int>::max()) {
            cout << "\nNo solution found meeting the criteria." << endl;
        } else {
            cout << "\nGlobal Minimum cut cost: " << global_min_info.cost
                 << " (found by rank " << global_min_info.rank << ")" << endl;

            if (global_min_info.rank == 0) {
                cout << "Best flags configuration (Rank 0): [ ";
                for(int flag : local_bestFlags) { cout << flag << " "; }
                cout << "]" << endl;
            } else {
                vector<int> final_bestFlags(V);
                MPI_Recv(final_bestFlags.data(), V, MPI_INT, global_min_info.rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cout << "Best flags configuration (Rank " << global_min_info.rank << "): [ ";
                for(int flag : final_bestFlags) { cout << flag << " "; }
                cout << "]" << endl;
            }
        }
        cout << "\nTotal Wall Time: " << (end_time - start_time) << " seconds" << endl;
        double max_local_time;
        double local_duration = local_end_time - local_start_time;
        MPI_Reduce(&local_duration, &max_local_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        cout << "Max Local Computation Time: " << max_local_time << " seconds" << endl;

    } else {
        if (world_rank == global_min_info.rank && global_min_info.cost != numeric_limits<int>::max()) {
             if (local_bestFlags.size() == (size_t)V) {
                MPI_Send(local_bestFlags.data(), V, MPI_INT, 0, 0, MPI_COMM_WORLD);
             } else {
                  cerr << "Rank " << world_rank << " Error: Cannot send best flags, size is incorrect (" << local_bestFlags.size() << " vs V=" << V << ")" << endl;
             }
        }
        double local_duration = local_end_time - local_start_time;
        MPI_Reduce(&local_duration, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}