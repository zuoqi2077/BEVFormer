
## ✅ BEVFormer：mmdet3d_plugin 注册冲突处理记录（force=True 批量修复）

### 背景 / 触发条件

在 `import mmdet3d_plugin` 或 `build model` 时出现类似报错：

* `KeyError: '<ClassName> is already registered in <registry>'`

原因（人话版）：

* `mmdet/mmdet3d/mmcv` 在 import 时已经注册了一些组件
* BEVFormer 的 `projects/mmdet3d_plugin` 也会在 import 时注册同名组件
* 默认 registry 不允许重名 → 直接抛异常，导致 plugin 无法 import / 模型无法 build

### 目标

* **不训练，只为能 import plugin + build model**
* 允许 plugin 覆盖同名注册项：把 `register_module()` 改为 `register_module(force=True)`

### 操作步骤（批量、可审计）

#### 0) 确认当前分支（避免污染 master）

```bash
git branch --show-current
# 期望：study-notes
```

#### 1) 统计将要修改的数量（不改文件）

```bash
grep -R --line-number --fixed-string "register_module()" projects/mmdet3d_plugin | wc -l
```

#### 2) 批量替换（仅替换空括号形式）

```bash
find projects/mmdet3d_plugin -name "*.py" -print0 \
| xargs -0 sed -i 's/register_module()/register_module(force=True)/g'
```

> 说明：只处理 `register_module()`（空括号），已有参数的注册（如 `register_module(name=...)`）不会被这条命令改到。

#### 3) 验收门禁（必须做）

1. 改动范围是否合理（不应该出现“整文件重写”的巨量 diff）

```bash
git diff --stat
```

2. plugin 是否能完整 import（关键门禁）

```bash
PYTHONPATH=$PWD/projects python -c "import mmdet3d_plugin; print('OK')"
# 期望输出：OK
```

### 回滚方式（本地不需要联网）

如果发现 diff 异常或替换误伤：

```bash
git restore .
```

### 结果判据（通过条件）

* `import mmdet3d_plugin` 输出 `OK`
* 后续 `build_bevformer_model.py` 不再因 registry 重名中断

### 备注 / 风险说明

* `force=True` 会让 plugin 版本覆盖框架默认注册项
* 对“学习阶段（build/debug，不训练）”是合理取舍
* 若未来要做严谨训练/复现，可能需要回到“逐点冲突定位 + 精准注册策略”