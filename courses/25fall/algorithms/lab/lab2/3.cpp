#include <iostream>
using namespace std;

const long long MOD = 1000000007;

struct Mat {
    long long a[2][2];
};

// 2x2 矩阵乘法，每一步都对 MOD 取模
Mat multiply(const Mat &x, const Mat &y) {
    Mat r;
    r.a[0][0] = ( (x.a[0][0]*y.a[0][0]) % MOD + (x.a[0][1]*y.a[1][0]) % MOD ) % MOD;
    r.a[0][1] = ( (x.a[0][0]*y.a[0][1]) % MOD + (x.a[0][1]*y.a[1][1]) % MOD ) % MOD;
    r.a[1][0] = ( (x.a[1][0]*y.a[0][0]) % MOD + (x.a[1][1]*y.a[1][0]) % MOD ) % MOD;
    r.a[1][1] = ( (x.a[1][0]*y.a[0][1]) % MOD + (x.a[1][1]*y.a[1][1]) % MOD ) % MOD;
    return r;
}

// 快速幂：A^n = A^(1/2*n) * A^(1/2*n) 
Mat quick_pow(Mat base, long long n) {
    Mat result = { { {1, 0}, {0, 1} } };  // 单位矩阵 I
    while (n > 0) {
        if (n & 1) result = multiply(result, base);
        base = multiply(base, base);
        n >>= 1;
    }
    return result;
}

int main() {
    //加快输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n;
    cin >> n;

    // 斐波那契矩阵
    Mat A = { { {1, 1}, {1, 0} } };

    Mat Ans = quick_pow(A, n);   // A^n
    long long Fn = Ans.a[0][1]; // A^n 的 (0,1) 恰好是 F_n
    cout << Fn % MOD << "\n";
    return 0;
}
