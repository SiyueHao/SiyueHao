#include <iostream>
#include <cmath>
using namespace std;
using int64 = long long;

int main(){
    //加快输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    //读入参数
    int64 N; 
    cin >> N;

    // 选择、插入、冒泡的算法最坏比较次数相等
    int64 result1 = N * (N - 1) / 2;

    // 归并排序比较次数
    double lg = log2((double)N);                
    int k = (int)ceil(lg); //向下取整
    int64 two_pow_k = 1LL << k;                 
    int64 result2 = N * 1LL * k - two_pow_k + 1;

    // 输出较小值
    cout << (result1 < result2 ? result1 : result2);
    return 0;
}
