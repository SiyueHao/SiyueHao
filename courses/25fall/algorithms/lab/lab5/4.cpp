#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>

class FlowNetwork {
private:

    struct Arc {
        int to;
        long long residual; // 剩余容量
    };

    int num_nodes;
    std::vector<Arc> edges;          // 存储所有边（正向+反向）
    std::vector<std::vector<int>> adj; // 邻接表只存边在 edges 中的下标
    std::vector<int> depth;          // BFS 分层图深度
    std::vector<int> cur_arc;        // 当前弧优化指针

    // 检查是否有增广路，并建立分层图
    bool build_layered_graph(int s, int t) {
        std::fill(depth.begin(), depth.end(), -1);
        std::queue<int> q;
        
        depth[s] = 0;
        q.push(s);

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (int idx : adj[u]) {
                int v = edges[idx].to;
                // 如果有残量且未访问
                if (edges[idx].residual > 0 && depth[v] == -1) {
                    depth[v] = depth[u] + 1;
                    q.push(v);
                }
            }
        }
        return depth[t] != -1;
    }

    // 沿分层图推送流量
    long long push_flow(int u, int t, long long flow_limit) {
        if (u == t || flow_limit == 0) return flow_limit;

        // 当前弧优化：从上次访问的边开始遍历
        for (int &i = cur_arc[u]; i < (int)adj[u].size(); ++i) {
            int edge_idx = adj[u][i];
            int v = edges[edge_idx].to;
            
            if (depth[v] == depth[u] + 1 && edges[edge_idx].residual > 0) {
                long long pushed = push_flow(v, t, std::min(flow_limit, edges[edge_idx].residual));
                
                if (pushed > 0) {
                    edges[edge_idx].residual -= pushed;
                    edges[edge_idx ^ 1].residual += pushed; // i^1 是 i 的反向边
                    return pushed;
                }
            }
        }
        return 0;
    }

public:
    explicit FlowNetwork(int n) {
        num_nodes = n;
        adj.resize(n + 1);
        depth.resize(n + 1);
        cur_arc.resize(n + 1);
    }

    // 添加单向容量
    void add_capacity(int u, int v, long long cap) {
        // 正向边下标为偶数 k
        adj[u].push_back(edges.size());
        edges.push_back({v, cap});

        // 反向边下标为奇数 k+1
        adj[v].push_back(edges.size());
        edges.push_back({u, 0});
    }

    long long compute_max_flow(int s, int t) {
        long long total_flow = 0;
        const long long INF_FLOW = std::numeric_limits<long long>::max();

        while (build_layered_graph(s, t)) {
            std::fill(cur_arc.begin(), cur_arc.end(), 0);
            while (long long sent = push_flow(s, t, INF_FLOW)) {
                total_flow += sent;
            }
        }
        return total_flow;
    }
};

int main() {
    // 加快输入输出
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int nodes, edges, start_node, end_node;
    if (std::cin >> nodes >> edges >> start_node >> end_node) {
        FlowNetwork solver(nodes);

        for (int i = 0; i < edges; ++i) {
            int u, v;
            long long w;
            std::cin >> u >> v >> w;
            solver.add_capacity(u, v, w);
        }

        std::cout << solver.compute_max_flow(start_node, end_node) << "\n";
    }

    return 0;
}