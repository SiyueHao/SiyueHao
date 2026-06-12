#include <iostream>
#include<vector>
using namespace std;

int main() {
    //加快输入输出速度
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int Pop, Ind;
    cin >> Pop >> Ind;

    vector<int> ind_of_pop(Pop);
    vector<int> num_of_ind(Ind, 0);

    //读入每个人所属的行业，并更新每个行业的人数
    for (int i = 0; i < Pop; ++i) {
        cin >> ind_of_pop[i];
        ind_of_pop[i]--;                 
        num_of_ind[ind_of_pop[i]]++;
    }

    //读入每个人的劝说代价
    vector<long long> cost_of_trans(Pop);
    for (int i = 0; i < Pop; ++i) cin >> cost_of_trans[i];

    // 统计空行业数量 k，即要填补的行业数
    int k = 0;
    for (int t = 0; t < Ind; ++t) if (num_of_ind[t] == 0) ++k;
    if (k == 0) { cout << 0 << '\n'; return 0; }

    // 贪心算法，对于每个行业，留下劝说代价最大的居民，剩下的都是候选人
    vector<long long> keep_max(Ind, LLONG_MIN); //保留下的每个城市的代价最大的居民
    vector<long long> movable;  // 候选人
    movable.reserve(Pop);

    for (int i = 0; i < Pop; ++i) {
        int t = ind_of_pop[i];
        long long c = cost_of_trans[i];
        if (keep_max[t] < c) {
            if (keep_max[t] != LLONG_MIN) movable.push_back(keep_max[t]);
            keep_max[t] = c;
        } else {
            movable.push_back(c);
        }
    }

    // 计算总劝说代价
    nth_element(movable.begin(), movable.begin() + k, movable.end());
    long long cost = 0;
    for (int i = 0; i < k; ++i) cost += movable[i];

    cout << cost;
    return 0;
}
