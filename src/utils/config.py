# src/utils/config.py
import yaml, argparse, types
def load_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.model: cfg["model"]["name"] = args.model
    if args.epochs: cfg["epochs"] = args.epochs
    return types.SimpleNamespace(**cfg, **cfg.get("runtime", {}))
