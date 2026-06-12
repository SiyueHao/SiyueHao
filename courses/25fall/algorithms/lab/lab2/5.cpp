#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

int main() {
    //加快输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    // 创建邻接表
    vector<vector<int> > g(n + 1);

    // 读入 n-1 条边，令u 是 v 的父亲
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    int x, y;
    cin >> x >> y;

    // 求解
    vector<int> depth(n + 1, 0);   // 根深度定义为 1
    vector<int> parent(n + 1, 0);  // 记录每个点的父亲（根的父亲记 0）
    vector<int> levelCount(n + 2, 0); // 统计每一层的节点数（层号=depth）

    queue<int> q;
    depth[1] = 1;  
    parent[1] = 0;
    q.push(1);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        levelCount[depth[u]]++; // 统计该层节点数
        for (int v : g[u]) {
            if (v == parent[u]) continue; // 不往回走
            parent[v] = u;
            depth[v] = depth[u] + 1;
            q.push(v);
        }
    }

    // 最大的 depth 值
    int maxDepth = 0;
    int maxWidth = 0;
    for (int d = 1; d <= n; ++d) {
        if (levelCount[d] == 0) break; // 没有节点了
        maxDepth = d;
        if (levelCount[d] > maxWidth) maxWidth = levelCount[d];
    }


   
    int LOG = 1;
    while ((1 << LOG) <= n) LOG++;

    vector<vector<int> > up(n + 1, vector<int>(LOG, 0));
    // 第 0 层（2^0=1 级祖先）就是 parent
    for (int u = 1; u <= n; ++u) up[u][0] = parent[u];
    // 其余层：up[u][k] = up[ up[u][k-1] ][k-1]
    for (int k = 1; k < LOG; ++k) {
        for (int u = 1; u <= n; ++u) {
            int mid = up[u][k - 1];
            up[u][k] = (mid == 0 ? 0 : up[mid][k - 1]);
        }
    }

    // 把 a 提升到与 b 同深度
    auto lift_to = [&](int a, int targetDepth) {
        int diff = depth[a] - targetDepth;
        for (int k = 0; k < LOG; ++k) {
            if (diff & (1 << k)) a = up[a][k];
        }
        return a;
    };

    // LCA 查询
    auto lca = [&](int a, int b) {
        if (depth[a] < depth[b]) swap(a, b);

        // 让 a 和 b 到同一深度
        a = lift_to(a, depth[b]);
        if (a == b) return a;

        //从高位往低位同时向上跳，直到它们的父亲相同
        for (int k = LOG - 1; k >= 0; --k) {
            if (up[a][k] != up[b][k]) {
                a = up[a][k];
                b = up[b][k];
            }
        }
        //此时 a、b 的父亲就是 LCA
        return parent[a];
    };

    int L = lca(x, y);

    // 按题意：x至LCA段每条边耗时为2；LCA至y段每条边耗时为1
    long long latency =
        2LL * (depth[x] - depth[L]) +
        1LL * (depth[y] - depth[L]);

   
    //输出结果
    cout << maxDepth << endl;
    cout << maxWidth << endl;
    cout << latency << endl;

    return 0;
}
