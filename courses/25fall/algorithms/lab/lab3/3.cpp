#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, k;
    if (!(cin >> n >> m >> k)) return 0;

    // 读入：从顶行到底行
    vector<vector<int>> score(n, vector<int>(m));
    vector<vector<char>> typec(n, vector<char>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cin >> score[i][j] >> typec[i][j];
        }
    }

    // 特判：k == 0 时，一枪都打不出去
    if (k == 0) {
        cout << 0 << '\n';
        return 0;
    }

    const long long NEG = (long long)-4e18;

    // dpY[j]：打了 j 个 N，且所有被攻击过的列都以 Y 结尾 的最大得分
    // dpN[j]：打了 j 个 N，且至少有一列以 N 结尾 的最大得分
    vector<long long> dpY(k + 1, NEG), dpN(k + 1, NEG);
    dpY[0] = 0;

    // 按列做“分组背包”
    for (int col = 0; col < m; ++col) {
        // 预处理这一列从底往上的前缀信息
        vector<long long> val(n + 1, 0);   // 前缀得分
        vector<int> cost(n + 1, 0);        // 前缀中 N 的数量
        vector<int> lastType(n + 1, 0);    // 0:无, 1:Y, 2:N

        long long sumv = 0;
        int cntN = 0;
        for (int t = 1; t <= n; ++t) {
            int r = n - t;  // 自底向上第 t 个，对应输入中的第 r 行
            sumv += score[r][col];
            if (typec[r][col] == 'N') cntN++;
            val[t] = sumv;
            cost[t] = cntN;
            lastType[t] = (typec[r][col] == 'Y') ? 1 : 2;
        }

        vector<long long> newY(k + 1, NEG), newN(k + 1, NEG);

        // 1) 不打该列（t = 0），属性不变
        for (int j = 0; j <= k; ++j) {
            if (dpY[j] != NEG) newY[j] = max(newY[j], dpY[j]);
            if (dpN[j] != NEG) newN[j] = max(newN[j], dpN[j]);
        }

        // 2) 选择前缀 t = 1..n
        for (int t = 1; t <= n; ++t) {
            int w = cost[t];
            long long v = val[t];
            if (w > k) continue;  // 这一个前缀就已经用掉 >k 个 N，不可能

            for (int j = 0; j + w <= k; ++j) {
                if (dpY[j] != NEG) {
                    if (lastType[t] == 1) {
                        // 新列也以 Y 结尾，仍然是“所有列以 Y 结尾”
                        newY[j + w] = max(newY[j + w], dpY[j] + v);
                    } else { // lastType[t] == 2
                        // 出现了第一列以 N 结尾
                        newN[j + w] = max(newN[j + w], dpY[j] + v);
                    }
                }
                if (dpN[j] != NEG) {
                    // 已经至少有一列以 N 结尾了，状态仍然是 dpN
                    newN[j + w] = max(newN[j + w], dpN[j] + v);
                }
            }
        }

        dpY.swap(newY);
        dpN.swap(newN);
    }

    long long ans = 0;

    // 情况 1：没有用满子弹（打的 N 少于 k），dpY 和 dpN 都合法
    for (int j = 0; j < k; ++j) {
        ans = max(ans, dpY[j]);
        ans = max(ans, dpN[j]);
    }

    // 情况 2：用满子弹（打了恰好 k 个 N），必须是“至少一列以 N 结尾”
    ans = max(ans, dpN[k]);

    cout << ans << '\n';
    return 0;
}
