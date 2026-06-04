# Git 使用教程 — 结合你的 AI Python 学习路线

> 配合 VSCode 使用，面向网络科学研究项目

---

## 一、为什么要用 Git？

你现在有 8 个阶段的 Python 代码（从基础语法到机器学习），每个阶段都会不断修改。Git 能帮你：

- **时光机**：改坏了代码？一键回到昨天的版本
- **备份**：代码推到 GitHub，电脑坏了也不丢
- **协作**：以后和导师、同学一起写论文代码，不冲突
- **记录**：每次改动都有说明，回头看知道为什么这样改

---

## 二、核心概念（3 分钟理解）

```
你的电脑（本地）              GitHub（云端）
┌─────────────┐              ┌─────────────┐
│  工作区      │  git add     │             │
│ (你写的代码)  │ ────────→   │             │
│              │              │             │
│  暂存区      │  git commit  │             │
│ (准备提交的)  │ ────────→   │  远程仓库    │
│              │              │             │
│  本地仓库    │  git push    │             │
│ (已提交的)    │ ────────→   │             │
└─────────────┘              └─────────────┘

git pull ← 从 GitHub 拉取最新代码到本地
```

**一句话总结**：改代码 → add → commit → push，就是这么简单。

---

## 三、VSCode 里怎么用 Git（主要方式）

### 3.1 提交代码（最常用）

1. 左侧活动栏点击「源代码管理」图标（分叉的那个）
2. 你会看到所有修改过的文件列表
3. 鼠标悬停文件名，点 `+` 号暂存（相当于 `git add`）
4. 或者直接点「暂存所有更改」（相当于 `git add .`）
5. 在上方输入框写提交说明，比如「完成 phase5 级联失效分析」
6. 点「提交」按钮 或按 `Ctrl+Enter`
7. 点「同步」按钮 或按 `Ctrl+Shift+P` → 输入 `Git: Push`

**VSCode 小技巧**：
- 打开文件后，修改的行会有绿色（新增）或蓝色（修改）标记
- 点击行号旁边的标记，可以看到具体改了什么
- 左侧文件名旁边的 `M` = 修改，`U` = 新文件，`D` = 删除

### 3.2 查看历史

- `Ctrl+Shift+P` → 输入 `Git: View History` → 看到所有提交记录
- 或者安装 GitLens 扩展，鼠标悬停任意一行代码就能看到谁改的、什么时候改的

### 3.3 推送到 GitHub

- 提交后，点左下角的「同步」图标（圆圈箭头）
- 或者 `Ctrl+Shift+P` → `Git: Push`

### 3.4 拉取最新代码

- `Ctrl+Shift+P` → `Git: Pull`
- 或者点左下角同步图标

---

## 四、常用 Git 命令（VSCode 终端里用）

打开终端：`Ctrl+` `（反引号键，在 Esc 下面）

### 4.1 日常三连（99% 的时间只用这三行）

```bash
git add .                              # 暂存所有改动
git commit -m "说明改了什么"             # 提交
git push                               # 推送到 GitHub
```

### 4.2 查看状态

```bash
git status          # 当前有哪些文件改了、新增了、删除了
git log --oneline   # 提交历史（简洁版）
git log --oneline --graph   # 带分支图的提交历史
git diff            # 具体改了哪些内容（未暂存的）
git diff --staged   # 暂存区里改了什么
```

### 4.3 分支操作（重要！）

分支就像平行宇宙，你可以在不影响主线的情况下尝试新功能。

```bash
git branch                      # 查看所有分支
git branch experiment1          # 创建新分支
git checkout experiment1        # 切换到新分支
# 或者简写：
git checkout -b experiment1     # 创建并切换

# 在 experiment1 分支上随便改、随便提交
git add . && git commit -m "尝试新的级联模型"

# 切回主分支
git checkout main

# 如果实验成功，合并到主分支
git merge experiment1

# 实验失败？直接删掉分支，啥影响都没有
git branch -d experiment1
```

### 4.4 撤销操作

```bash
# 改了文件但还没 add，想丢弃修改
git checkout -- 文件名.py

# 已经 add 了但还没 commit，想取消暂存
git reset HEAD 文件名.py

# 已经 commit 了但还没 push，想修改最后一次提交
git commit --amend -m "新的提交说明"

# 已经 commit 了，想回到上一个版本（保留修改）
git reset --soft HEAD~1

# 危险！彻底丢弃最近一次提交的修改
git reset --hard HEAD~1
```

### 4.5 .gitignore 你已经在用了

你的项目已经有 `.gitignore`，这些文件会被自动忽略：
- `__pycache__/` — Python 缓存
- `.venv/` — 虚拟环境
- `.idea/` — PyCharm 配置
- `.vscode/` — VSCode 配置
- `.claude/` — Claude Code 配置
- `.pytest_cache/` — 测试缓存

如果想忽略更多文件，编辑 `.gitignore` 即可。

---

## 五、结合你的学习路线的工作流

你有 8 个阶段，推荐这样用 Git：

### 场景1：学习新阶段

```bash
# 开始 phase5 级联失效
git checkout -b phase5-cascading

# 写代码、测试、修改...
git add .
git commit -m "phase5: 基础级联失效模型"
git commit -m "phase5: 加入节点容量参数"
git commit -m "phase5: 完成级联可视化"

# 阶段完成，合并回主分支
git checkout main
git merge phase5-cascading
git push
```

### 场景2：对比不同阶段的代码

```bash
git log --oneline --all --graph   # 看整体提交历史
git diff phase1..phase5           # 对比 phase1 和 phase5 的代码差异
```

### 场景3：代码改坏了想回退

```bash
git log --oneline          # 找到想回到的那个版本号
git checkout abc1234       # 临时查看（只读）
git checkout main          # 回到当前
git revert abc1234         # 安全回退（创建一个新的撤销提交）
```

---

## 六、推荐安装的 VSCode 扩展

在 VSCode 扩展商店搜索安装：

| 扩展名 | 作用 |
|--------|------|
| **GitLens** | 悬停查看每行代码的作者和修改时间，超强 |
| **Git Graph** | 可视化分支图，直观看到提交历史 |
| **GitHub Pull Requests** | 在 VSCode 里直接创建 PR |

安装方法：`Ctrl+Shift+X` → 搜索扩展名 → 安装

---

## 七、提交信息规范

好的提交信息能让你三个月后还能看懂。推荐格式：

```
类型: 简短描述

phase1: 完成变量和数据类型练习
phase3: 新增四种网络模型实现
fix: 修复 BA 模型度分布计算错误
refactor: 重构网络可视化模块
docs: 更新 README 使用说明
```

类型说明：
- `feat` — 新功能
- `fix` — 修复 bug
- `refactor` — 重构代码（不改变功能）
- `docs` — 文档更新
- `test` — 测试相关

---

## 八、常见问题

**Q: push 时提示要输入用户名密码怎么办？**
A: 你已经配了 SSH 密钥，确保用的是 SSH 地址（`git@github.com:...`），不是 HTTPS 地址。
检查方式：`git remote -v`，如果显示 https，改成 ssh：
```bash
git remote set-url origin git@github.com:like11ey/ai_python.git
```

**Q: 不小心把 __pycache__ 提交了怎么办？**
A: 你已经有 .gitignore 了，以后不会了。要清除已提交的：
```bash
git rm -r --cached __pycache__ */__pycache__
git commit -m "chore: 移除缓存文件"
```

**Q: 两个人同时改了同一个文件怎么办？**
A: Git 会提示冲突，VSCode 会高亮显示冲突位置，你选择保留哪个版本就行。

**Q: 我想保存当前进度但还没写完怎么办？**
```bash
git stash              # 临时保存
# 去做别的事...
git stash pop          # 恢复之前保存的
```

---

## 九、你的 Git 配置总览

```
用户名:   Like11ey
邮箱:     1537094320@qq.com
认证:     SSH 密钥 (ed25519)
默认分支: main
代理:     127.0.0.1:7897
```

---

> 💡 **最简记忆法**：改代码 → `git add .` → `git commit -m "xxx"` → `git push`
> VSCode 里就点三个按钮：暂存 → 提交 → 同步
