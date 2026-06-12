#include <stdio.h>

#define MAXN 205
#define MAXM 205
#define MAXK 205

const long long NEG = -(1LL << 60);

int main(void) {
    int n, m, k;
    if (scanf("%d %d %d", &n, &m, &k) != 3) return 0;

    int score[MAXN][MAXM];
    char type[MAXN][MAXM];

    // 输入：从“最上行”到“最下行”
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int f;
            char c;
            scanf("%d %c", &f, &c);
            score[i][j] = f;
            type[i][j] = c; // 'N' 或 'Y'
        }
    }

    // 初始鱼雷为 0，连第一发都打不出去
    if (k == 0) {
        printf("0\n");
        return 0;
    }

    // dp[used][flag]
    // used: 已经实际消耗的 N 型鱼雷数（0..k）
    // flag: 是否存在某一列被选择的前缀最后一架是 N（0/1）
    static long long dp[MAXK][2], nxt[MAXK][2];

    for (int u = 0; u <= k; ++u)
        dp[u][0] = dp[u][1] = NEG;
    dp[0][0] = 0;  // 还没选任何列

    // 逐列做“分组背包”
    for (int col = 0; col < m; ++col) {
        // 预处理当前列所有可能前缀：
        // 打掉底部 t 架 (t = 0..n) 时的得分 val[t]、消耗的 N 数量 cost[t]、
        // 以及该前缀最后一架是否是 N（lastIsN[t]）
        int cost[MAXN];
        long long val[MAXN];
        int lastIsN[MAXN];

        cost[0] = 0;
        val[0] = 0;
        lastIsN[0] = 0;  // 不打任何飞机

        int cntN = 0;
        long long sumS = 0;

        for (int t = 1; t <= n; ++t) {
            int row = n - t;                  // 底部往上数第 t 架
            sumS += score[row][col];
            if (type[row][col] == 'N') cntN++;
            cost[t] = cntN;
            val[t]  = sumS;
            lastIsN[t] = (type[row][col] == 'N');  // 前缀最后一架就是当前这架
        }

        // 初始化下一轮 dp
        for (int u = 0; u <= k; ++u)
            nxt[u][0] = nxt[u][1] = NEG;

        // 分组转移
        for (int used = 0; used <= k; ++used) {
            for (int f = 0; f <= 1; ++f) {
                if (dp[used][f] == NEG) continue;

                // 在本列选择前缀 t（0..n）
                for (int t = 0; t <= n; ++t) {
                    int c = cost[t];
                    int newUsed = used + c;
                    if (newUsed > k) continue;

                    int nf = f || lastIsN[t];
                    long long cand = dp[used][f] + val[t];
                    if (cand > nxt[newUsed][nf])
                        nxt[newUsed][nf] = cand;
                }
            }
        }

        // 更新 dp
        for (int u = 0; u <= k; ++u) {
            dp[u][0] = nxt[u][0];
            dp[u][1] = nxt[u][1];
        }
    }

    long long ans = 0;

    // 情况一：实际消耗的 N 数量 <= k-1，任何 flag 都合法
    for (int used = 0; used <= k - 1; ++used) {
        for (int f = 0; f <= 1; ++f) {
            if (dp[used][f] > ans) ans = dp[used][f];
        }
    }

    // 情况二：实际消耗刚好 k 发，需要存在“前缀以 N 结尾”的列
    if (dp[k][1] > ans) ans = dp[k][1];

    printf("%lld\n", ans);
    return 0;
}
