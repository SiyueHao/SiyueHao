#include <iostream>
#include <vector>
#include <algorithm>   // 新增：使用 nth_element
using namespace std;


// 原冒泡排序函数（使用会超时）
// void BubbleSort(vector<int>& a, int N) {
//     for (int i = 0; i < N - 1; i++) {
//         for (int j = 0; j < N - 1 - i; j++) {
//             if (a[j] > a[j + 1]) swap(a[j], a[j + 1]);
//         }
//     }
// }

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 读入参数
    long long sum = 0;   
    int N;
    cin >> N;
    vector<long long> a(N);   

    for (int i = 0; i < N; i++) {
        cin >> a[i];
        sum += a[i];
    }

    //只需要中位数
    nth_element(a.begin(), a.begin() + N / 2, a.end());

    // 获取中位数
    long long middle = a[N / 2];

    // 计算最小下界
    long long min_sum = middle * 2LL * N + 1;  
    long long x = min_sum - sum;

    // 输出结果
    if (x >= 0)
        cout << x;
    else
        cout << -1;

    return 0;
}
