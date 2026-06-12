# include<iostream>
# include<vector>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    if (!(cin >> N)) return 0;

    // 读入每个节点的魔法值
    vector<long long> a(N + 1);
    for (int i = 1; i <= N; ++i) cin >> a[i];

    // 建树：无向图邻接表
    vector<vector<int>> g(N + 1);
    for (int k = 0; k < N - 1; ++k) {
        int u, v; 
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    //确定parent
    // parent[u] = 父亲；根 1 的 parent 记为 0
    vector<int> parent(N + 1, 0);
    vector<int> order; 
    order.reserve(N);

    // 用栈模拟 DFS
    stack<int> st;
    st.push(1);
    parent[1] = 0;

    while (!st.empty()) {
        int u = st.top(); 
        st.pop();
        order.push_back(u);              // 记录访问到的顺序（前序）
        // 遍历邻居
        for (int v : g[u]) {
            if (v == parent[u]) continue; // 别走回头路
            parent[v] = u;                
            st.push(v);                   
        }
    }

   //反向遍历order
    vector<long long> subsum(N + 1, 0);   // subsum[u] = 以 u 为根的子树魔法值总和
    long long best = LLONG_MIN;           

    for (int i = (int)order.size() - 1; i >= 0; --i) {
        int u = order[i];
        long long s = a[u];               
        // 把所有孩子的子树和加上来（孩子的 parent 是 u）
        for (int v : g[u]) {
            if (v == parent[u]) continue; // 跳过父亲
            s += subsum[v];
        }
        subsum[u] = s;                   
        if (s > best) best = s;          
    }

    //输出
    cout << best << "\n";
    return 0;
}
