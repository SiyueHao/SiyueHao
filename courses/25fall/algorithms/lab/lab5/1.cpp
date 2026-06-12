#include <iostream>
#include <vector>
#include <string>

void run_solver() {
    int len;
    std::cin >> len;
    
    std::string text;
    std::cin >> text;

    // fail[i] 对应原代码的 pi[i]，表示前缀长度为 i 时的最长 border 长度
    // min_border[i] 对应原代码的 sb[i]，表示最短非空 border 长度
    std::vector<int> fail(len + 1, 0);
    std::vector<int> min_border(len + 1, 0);

    long long total_period_len = 0;

    // 计算 fail 指针和最短 border
    for (int i = 2, j = 0; i <= len; ++i) {
        // KMP 匹配过程
        while (j > 0 && text[i - 1] != text[j]) {
            j = fail[j];
        }
        if (text[i - 1] == text[j]) {
            j++;
        }
        
        fail[i] = j;

        // 动态规划计算最短 border
        if (j == 0) {
            min_border[i] = 0;
        } else {
            // 如果 fail[i] 指向的位置本身还有 border，则继承它的最短 border
            // 否则 fail[i] 本身就是最短 border
            if (min_border[j] != 0) {
                min_border[i] = min_border[j];
            } else {
                min_border[i] = j;
            }
        }

        // 累加结果：最大周期 = 当前长度 - 最短 border
        if (min_border[i] > 0) {
            total_period_len += (i - min_border[i]);
        }
    }

    std::cout << total_period_len << "\n";
}

int main() {
    // 加快输入输出
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    run_solver();

    return 0;
}