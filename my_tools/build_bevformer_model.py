import argparse
import importlib
import os
import sys

import torch


def try_import_plugin(cfg, repo_root: str):
    """BEVFormer 用 plugin_dir 把 projects/mmdet3d_plugin 注入 registry。"""
    if not cfg.get("plugin", False):
        return

    plugin_dir = cfg.get("plugin_dir", None)
    if plugin_dir is None:
        raise RuntimeError("cfg.plugin=True 但找不到 cfg.plugin_dir")

    # plugin_dir 在 config 里通常是相对路径，比如 'projects/mmdet3d_plugin/'
    plugin_path = plugin_dir
    if not os.path.isabs(plugin_path):
        plugin_path = os.path.join(repo_root, plugin_dir)

    if not os.path.isdir(plugin_path):
        raise RuntimeError(f"plugin_dir 路径不存在: {plugin_path}")

    # 把 plugin_path 加进 sys.path，然后 import mmdet3d_plugin
    if plugin_path not in sys.path:
        sys.path.insert(0, plugin_path)

    # 关键：触发 projects/mmdet3d_plugin/__init__.py，完成注册
    importlib.import_module("mmdet3d_plugin")


def build_model_from_cfg(cfg):
    """兼容 mmdet3d 常见 build 入口（不同版本命名略不同）。"""
    try:
        from mmdet3d.models import build_model  # 有些版本叫 build_model
        return build_model(cfg.model)
    except Exception:
        from mmdet3d.models import build_detector  # 老/常见入口
        return build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config 路径，例如 projects/configs/bevformer/bevformer_base.py")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--print-top", type=int, default=60, help="打印 model 前 N 行结构")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.abspath(args.config)

    # mmcv Config
    from mmcv import Config
    cfg = Config.fromfile(config_path)

    # 注入 plugin（把 BEVFormer 的自定义模块注册到 mmdet/mmdet3d 的 registry）
    try_import_plugin(cfg, repo_root)

    print("=" * 80)
    print(f"[Config] {config_path}")
    print(f"[Model type] {cfg.model.get('type', 'N/A')}")
    print("=" * 80)

    model = build_model_from_cfg(cfg)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()

    n_params = count_params(model)
    print(f"[Params] {n_params/1e6:.2f} M")

    # 打印结构（只打印前 N 行，防止刷屏）
    model_str = str(model).splitlines()
    print("=" * 80)
    print("[Model structure head]")
    for line in model_str[: args.print_top]:
        print(line)
    if len(model_str) > args.print_top:
        print(f"... (total {len(model_str)} lines)")
    print("=" * 80)

    # 可选：尝试 forward_dummy（如果作者实现了）
    # 这一步“不保证成功”，失败也正常，因为 detector 常需要 img_metas / 多相机输入等
    if hasattr(model, "forward_dummy"):
        try:
            print("[Try] model.forward_dummy(...)")
            dummy = torch.randn(1, 3, 256, 704, device=device)  # 随便给一个尺寸
            out = model.forward_dummy(dummy)
            print("[OK] forward_dummy returned type:", type(out))
        except Exception as e:
            print("[SKIP] forward_dummy failed (expected for many detectors):", repr(e))
    else:
        print("[INFO] model.forward_dummy 不存在，跳过最小 forward。")

    print("[DONE] build 成功")


if __name__ == "__main__":
    main()

