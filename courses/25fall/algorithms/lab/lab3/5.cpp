#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Interval {
    long long l, r;
};

int main() {
    //加快输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<Interval> segs(n);
    for (int i = 0; i < n; ++i) cin >> segs[i].l >> segs[i].r;

    sort(segs.begin(), segs.end(), [](const Interval &a, const Interval &b) {
        if (a.r != b.r) return a.r < b.r;
        return a.l < b.l;
    });

    long long cur_end = -4e18;
    int ans = 0;
    for (const auto &it : segs) {
        if (it.l >= cur_end) {
            ++ans;
            cur_end = it.r;
        }
    }

    cout << ans << '\n';
    return 0;
}
