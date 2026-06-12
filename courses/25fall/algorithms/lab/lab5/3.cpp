#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

class Solution {
private:
    static constexpr int LIMIT = 1000005;
    std::vector<int> lp;      // lowest prime factor
    std::vector<int> primes;

    // 预处理：线性筛
    void init_sieve() {
        lp.assign(LIMIT, 0);
        primes.reserve(LIMIT / 10);
        
        for (int i = 2; i < LIMIT; ++i) {
            if (lp[i] == 0) {
                lp[i] = i;
                primes.push_back(i);
            }
            for (int p : primes) {
                if (p > lp[i] || (long long)i * p >= LIMIT) break;
                lp[i * p] = p;
            }
        }
    }

    // 计算 square-free part 
    int get_core_val(int val) {
        int core = 1;
        while (val > 1) {
            int p = lp[val];
            int count = 0;
            while (val % p == 0) {
                val /= p;
                count++;
            }
            // 只有奇数次幂的质因子保留
            if (count & 1) core *= p;
        }
        return core;
    }

public:
    Solution() {
        init_sieve();
    }

    void solve() {
        int n;
        if (!(std::cin >> n)) return;

        std::vector<int> arr(n);
        // 使用 map 聚合冲突组：key 为质数 p
        // value.first: 频率为 p 的质数个数
        // value.second: kernel 为 p 的合数个数
        std::map<int, std::pair<int, int>> conflicts;

        for (int i = 0; i < n; ++i) {
            std::cin >> arr[i];
            int val = arr[i];

            if (val == 1) continue; // 1 不参与任何冲突

            if (lp[val] == val) {
                // 情况A: val 本身是质数
                conflicts[val].first++;
            } else {
                // 情况B: val 是合数，计算其核心
                int core = get_core_val(val);
                // 只有当核心也是质数时，才可能与该质数形成冲突
                if (core > 1 && lp[core] == core) {
                    conflicts[core].second++;
                }
            }
        }

        // 计算最大匹配数
        long long matching_pairs = 0;
        for (auto const& [prime_base, counts] : conflicts) {
            // 实际上是求二分图最大匹配
            matching_pairs += std::min(counts.first, counts.second);
        }

        // 最大独立集 = 总点数 - 最大匹配数
        std::cout << (long long)n - matching_pairs << "\n";
    }
};

int main() {
    // 加快输入输出
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    Solution solver;
    solver.solve();

    return 0;
}