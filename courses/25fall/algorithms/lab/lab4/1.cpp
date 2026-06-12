 #include <iostream>
#include <vector>
#include <queue>
#include <functional> // for greater

using namespace std;

// 定义邻接表存储图： pair<权重, 目标点>
//为了方便优先队列排序，通常把权重放在 pair 的第一位
typedef pair<int, int> PII;

int main() {
    // 1. 优化输入输出效率
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    // 特殊情况判断
    if (n == 1) {
        cout << 0 << endl;
        return 0;
    }

    // 2. 构建邻接表
    // adj[u] 包含所有从 u 出发的边 {权重, v}
    vector<vector<PII>> adj(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        if (u == v) continue; // 忽略自环
        // 无向图，双向添加
        adj[u].push_back({w, v});
        adj[v].push_back({w, u});
    }

    // 3. Prim 算法初始化
    long long total_weight = 0;
    int edges_count = 0;
    
    // visited 数组记录节点是否已加入生成树
    vector<bool> visited(n + 1, false);
    
    // 小顶堆：自动弹出权重最小的边
    // 存储格式：{权重, 节点}，表示到达该节点的边的权重
    priority_queue<PII, vector<PII>, greater<PII>> pq;

    // 从节点 1 开始
    pq.push({0, 1});

    while (!pq.empty()) {
        auto [w, u] = pq.top();
        pq.pop();

        // 如果节点 u 已经在生成树中，跳过
        if (visited[u]) continue;

        // 将节点 u 加入生成树
        visited[u] = true;
        total_weight += w;
        if (u != 1) edges_count++; // 起点不算一条边

        // 扫描 u 的所有邻居
        for (auto& edge : adj[u]) {
            int weight = edge.first;
            int v = edge.second;
            if (!visited[v]) {
                pq.push({weight, v});
            }
        }
    }

    // 4. 检查是否连通
    if (edges_count == n - 1) {
        cout << total_weight << "\n";
    } else {
        cout << -1 << "\n";
    }

    return 0;
}