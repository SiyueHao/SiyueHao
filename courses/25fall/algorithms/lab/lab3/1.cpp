//#include <bits/stdc++.h>
#include <iostream>
using namespace std;

const long long INF = (1LL << 60);  // 一个足够大的数

int main() {
    //加快输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    vector<long long>m(N+1);
    vector<long long> pre(N+1,0);//前缀和
    
    cin>>N;

    for(int i = 1 ; i <= N;i++){
        cin>>m[i];
        pre[i]= pre[i-1]+m[i];
    }
    
    long long dp[500][500];//题目给了N最大为400

    for (int len = 2; len <= N; ++len) {
        // l 是区间左端点
        for (int l = 1; l + len - 1 <= N; ++l) {
            int r = l + len - 1;
            dp[l][r] = INF;

            // 枚举最后一次合并的位置 k
            for (int k = l; k < r; ++k) {
                long long leftCost  = dp[l][k];
                long long rightCost = dp[k + 1][r];
                long long sumLR = pre[r] - pre[l - 1]; // [l, r] 的厚度总和

                long long cost = leftCost + rightCost + sumLR;
                if (cost < dp[l][r]) {
                    dp[l][r] = cost;
                }
            }
        }


    }

    cout<< dp[1][N]<<"\n";

    return 0;
}