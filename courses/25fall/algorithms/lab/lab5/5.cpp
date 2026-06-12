#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// 使用 DFS 进行连通分量搜索
class GridSolver {
private:
    int n;
    std::vector<std::string> board;
    std::vector<std::vector<int>> memo; // 直接存储每个位置能到达的方格数
    std::vector<std::vector<bool>> visited;

    // 方向数组
    const int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    struct Coord {
        int r, c;
    };

    // 深度优先搜索，收集同一个连通块内的所有点
    void dfs(int r, int c, std::vector<Coord>& component_nodes) {
        visited[r][c] = true;
        component_nodes.push_back({r, c});

        for (auto& d : dirs) {
            int nr = r + d[0];
            int nc = c + d[1];

            // 边界检查 + 访问检查
            if (nr >= 0 && nr < n && nc >= 0 && nc < n && !visited[nr][nc]) {
                // 题目核心逻辑：字符必须不同（0 <-> 1）
                if (board[nr][nc] != board[r][c]) {
                    dfs(nr, nc, component_nodes);
                }
            }
        }
    }

public:
    void solve() {
        int m;
        if (!(std::cin >> n >> m)) return;

        board.resize(n);
        for (int i = 0; i < n; ++i) {
            std::cin >> board[i];
        }

        // 初始化状态数组
        memo.assign(n, std::vector<int>(n, 0));
        visited.assign(n, std::vector<bool>(n, false));

        // 预处理所有连通块
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!visited[i][j]) {
                    // 记录当前连通块包含的所有坐标
                    std::vector<Coord> nodes;
                    dfs(i, j, nodes);

                    // 批量更新结果：该连通块内所有点的可达数量都等于连通块大小
                    int component_size = nodes.size();
                    for (const auto& p : nodes) {
                        memo[p.r][p.c] = component_size;
                    }
                }
            }
        }

        // 处理查询 
        while (m--) {
            int r, c;
            std::cin >> r >> c;
            // 题目输入通常是 1-based，转换为 0-based
            std::cout << memo[r - 1][c - 1] << "\n";
        }
    }
};

int main() {
    // 加快输入输出
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    GridSolver app;
    app.solve();

    return 0;
}