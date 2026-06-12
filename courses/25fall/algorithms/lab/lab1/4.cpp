#include<iostream>
#include<vector>
using namespace std;

int main(){

    //加快输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    //读入参数
    int N ;
    cin >> N ;

    int tem = 0;
    vector<int> a(N, 0);   // 把频次数组初始化为 0

    for (int i = 0; i < N; i++) {
        cin >> tem;
        a[tem - 1] ++;
    }
    
    long long S = 0; // 出现一次“频次为1”的人数
    long long E = 0; // 出现≥3次时多出来必须改号的人数之和

    for (int i = 0; i < N; i++) {
        if (a[i] == 1) S += 1;
        else if (a[i] > 2) E += (a[i] - 2);
    }

    // 若多余人数 E 足以覆盖所有频次为1的人数 S，答案就是 E；否则还需把剩余 (S - E) 个两两配对，每对至少改 1 人：再加 (S - E) / 2
    long long sum = (E >= S) ? E : (E + (S - E) / 2);

    cout << sum;
    return 0;
}
