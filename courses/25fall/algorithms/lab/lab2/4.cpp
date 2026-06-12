#include <iostream>
using namespace std;

string inorder, preorder, postorder;
int pos[500];  

// 使用此函数，通过给定的前序和中序遍历来构建后序遍历序列
void build(int preL, int preR, int inL, int inR) { //preL, preR表示前序遍历的左右边界，inL, inR表示中序遍历的左右边界
    if (preL > preR) return; // 递归边界条件
    char root = preorder[preL]; 
    int k = pos[(unsigned char)root];
    int L = k - inL; // 左子树大小
    build(preL + 1, preL + L, inL, k - 1); // 左
    build(preL + L + 1, preR, k + 1, inR); // 右
    postorder.push_back(root); // 根
}

int main() {
    //加快输入输出
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    //读入中序和前序遍历结果
    cin >> inorder >> preorder;
    int n = (int)inorder.size();
    for (int i = 0; i < n; ++i) pos[(unsigned char)inorder[i]] = i;
    build(0, n - 1, 0, n - 1);
    cout << postorder;
    return 0;
}