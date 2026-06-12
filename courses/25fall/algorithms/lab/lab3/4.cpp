#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

// 将 __int128 转为十进制字符串
static string toString(__int128 x) {
    if (x == 0) return "0";
    string s;
    bool neg = x < 0;
    if (neg) x = -x;
    while (x) {
        int d = int(x % 10);
        s.push_back(char('0' + d));
        x /= 10;
    }
    if (neg) s.push_back('-');
    reverse(s.begin(), s.end());
    return s;
}

int main() {
    //加快输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, k;
    if (!(cin >> n >> k)) return 0;

    vector<long long> w(n);
    for (int i = 0; i < n; ++i) cin >> w[i];

    if (n == 1) {
        cout << 0 << '\n';  // 单一符号可用空串表示
        return 0;
    }

    priority_queue<long long, vector<long long>, greater<long long>> pq;
    for (long long x : w) pq.push(x);

    // 补零使 (n-1) % (k-1) == 0
    int rem = (k - 1 - (n - 1) % (k - 1)) % (k - 1);
    for (int i = 0; i < rem; ++i) pq.push(0);

    __int128 ans = 0;
    while ((int)pq.size() > 1) {
        long long sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += pq.top();
            pq.pop();
        }
        ans += sum;
        pq.push(sum);
    }

    cout << toString(ans) << '\n';
    return 0;
}
