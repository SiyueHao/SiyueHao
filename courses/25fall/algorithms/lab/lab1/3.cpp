#include<iostream>
#include<vector>
using namespace std;
 
int main (){
    //加快输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    //读入参数
    long long N ;
    cin >> N ;

    vector<long long> a(N);  
    for (int i = 0; i < N; i++) {
        cin >> a[i];
    }

    // 遍历找最大值
    long long max = -1e18; //初始设为一个极小值
    long long cur  = 0;

    //最优子段和思想： 当前位置的最佳连续和要么是续上之前开始的最佳连续和，要么是从当前位置重新开始
    for ( int i = 0 ; i < N ; i ++){
        cur = (cur + a[i] >= a[i])? cur + a[i] : a[i];
        max = (max >= cur) ? max : cur;
    }
    cout << max ;
}