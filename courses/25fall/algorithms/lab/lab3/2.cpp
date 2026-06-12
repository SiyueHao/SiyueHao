#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
using namespace std;

// 线段树，支持区间最小值查询与单点更新
struct SegTree {
    int n; // 最大下标
    vector<long long> t;
    static constexpr long long INF = (1LL << 62);
    explicit SegTree(int n) : n(n), t((n + 1) << 2, INF) {}
    void update(int idx, long long val, int p, int l, int r) {
        if (l == r) {
            t[p] = val;
            return;
        }
        int m = (l + r) >> 1;
        if (idx <= m) update(idx, val, p << 1, l, m);
        else update(idx, val, p << 1 | 1, m + 1, r);
        t[p] = min(t[p << 1], t[p << 1 | 1]);
    }
    void update(int idx, long long val) { update(idx, val, 1, 0, n); }
    long long query(int L, int R, int p, int l, int r) const {
        if (L > r || R < l) return INF;
        if (L <= l && r <= R) return t[p];
        int m = (l + r) >> 1;
        return min(query(L, R, p << 1, l, m), query(L, R, p << 1 | 1, m + 1, r));
    }
    long long query(int L, int R) const {
        if (L > R) return INF;
        return query(L, R, 1, 0, n);
    }
};
constexpr long long SegTree::INF;

// 判断在限制值 X 下是否存在方案使总代价不超过 X
static bool feasible(const vector<long long> &a, const vector<long long> &pre, long long X) {
    int n = (int)a.size() - 1;  // a 从 1 开始

    const long long INF = (1LL << 62);
    vector<long long> dp_del(n + 1, INF);   // 前 i 个，且位置 i 被删除的最小删除和
    vector<long long> dp_keep(n + 1, INF);  // 前 i 个，位置 i 被保留的最小删除和

    dp_del[0] = 0;
    SegTree st(n);
    st.update(0, dp_del[0]);

    for (int i = 1; i <= n; ++i) {
        // 删除位置 i
        dp_del[i] = min(dp_del[i - 1], dp_keep[i - 1]) + a[i];
        st.update(i, dp_del[i]);

        // 选择一个以 i 结尾的保留段 [j, i]，段和需 ≤ X，且 j-1 必须被删除
        long long need = pre[i] - X;  // pre[j-1] 需要 >= need
        int idx = (int)(lower_bound(pre.begin(), pre.begin() + i, need) - pre.begin()); // j-1 的下界
        if (idx <= i - 1) {
            long long best = st.query(idx, i - 1);
            dp_keep[i] = best;
        }
    }

    long long removed = min(dp_del[n], dp_keep[n]);
    return removed <= X;
}

int main() {
    //加快输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n + 1), pre(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        cin >> a[i];
        pre[i] = pre[i - 1] + a[i];
    }
    long long total = pre[n];

    long long l = 0, r = total;         // 二分最小可行总代价
    while (l < r) {
        long long mid = l + (r - l) / 2;
        if (feasible(a, pre, mid)) r = mid;
        else l = mid + 1;
    }

    cout << l << '\n';
    return 0;
}
