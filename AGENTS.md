# ~/papers/ — 坍缩1号的论文精读笔记

## 这是什么

基于 Hugo + PaperMod 的论文笔记网站。源文件在 `content/papers/`，每篇一个 markdown 文件，`hugo build` 输出到 `/var/www/papers/`。

## 重要：git + post-commit hook

整个 `~/papers/`（含 content、static、layouts 等）已用 git 管理。**每个 commit 会自动触发 hugo build**，通过 `.git/hooks/post-commit` 实现：

```bash
#!/bin/bash
cd "$(git rev-parse --show-toplevel)" && hugo
```

### 工作流

1. 新增/修改笔记 → `content/papers/` 下的 markdown
2. `git add` + `git commit` → hook 自动跑 hugo → 站点自动更新
3. **不需要手动跑 `hugo`**

### 规范

- **slug 命名**: `{arxiv-id}-{short-kebab-title}` 如 `2606.03540-attend-to-anything-aam`
  - 非 arxiv 论文可自定义，如 `mit-kalman-filter-tutorial-lacey`
- **front matter**: 必须包含 `title`, `date`, `description`, `tldr`, `tags`, `slug`
- **构建命令**: `cd ~/papers && hugo`（hook 会自动做，一般不需要手动跑）

## 排除的文件（.gitignore）

- `public/` — 构建输出
- `.hugo_build.lock` — 锁文件
- `tinyhttpd` — 静态文件服务器二进制

## 主题

- PaperMod（submodule）
- minimal（备选）

## 相关

- TOOLS.md: Hugo Paper Notes 节有速查
- MEMORY.md: 大事记
