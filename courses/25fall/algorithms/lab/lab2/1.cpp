# include<iostream>
# include<vector>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    if (!(cin >> N)) return 0;
    vector<long long> A(N);
    for (int i = 0; i < N; ++i) cin >> A[i];

    // 按层检查是否回文（包含 -1，占位也参与比较）
    for (long long level = 0; ; ++level) {
        long long s = (1LL << level) - 1;                 // 本层起始下标
        if (s >= N) break;                                // 没有更多层
        long long t = min<long long>(N - 1, (1LL << (level + 1)) - 2); // 本层结束下标（截断到 N-1）
        long long len = t - s + 1;

        for (long long j = 0; j < len / 2; ++j) {
            if (A[s + j] != A[s + len - 1 - j]) {
                cout << "No";
                return 0;
            }
        }
    }
    cout << "Yes";
    return 0;
}
