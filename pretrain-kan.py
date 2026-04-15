import argparse
import importlib.util
import numpy as np
from pathlib import Path

import demo
from problem.kan import KAN


def load_module(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nda = load_module("nsga-nda.py", "nsga_nda_module")


def serialize_kan_model(model) -> dict:
    return {
        "state_dict": model.state_dict(),
        "config": {
            "width": list(model.width),
            "grid": int(model.grid),
            "k": int(model.k),
            "mult_arity": model.mult_arity,
            "base_fun": model.base_fun_name,
            "symbolic_enabled": bool(model.symbolic_enabled),
            "affine_trainable": bool(model.affine_trainable),
            "grid_eps": float(model.grid_eps),
            "grid_range": list(model.grid_range),
            "sp_trainable": bool(model.sp_trainable),
            "sb_trainable": bool(model.sb_trainable),
            "seed": int(getattr(model, "seed", 1)),
            "save_act": bool(model.save_act),
            "sparse_init": bool(model.sparse_init),
        },
    }


def pretrain_and_save_kan_surrogates(args):
    problems = ["ZDT1", "ZDT2", "ZDT3", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
    dims = [15, 20, 25, 30]  # Pretrain for multiple dimensions

    for problem_name in problems:
        for dim in dims:
            print(f"Pre-training KAN surrogate on {problem_name}-{dim}D...")
            problem = nda.ZDTProblem(name=problem_name, dim=dim)
            x_data, y_data, models = nda.pre_train_kan_surrogate_for_problem(
                problem=problem,
                device=args.device,
                kan_steps=args.kan_steps,
                hidden_width=args.kan_hidden,
                grid=args.kan_grid,
                seed=args.seed,
            )

            # Save the models
            save_path = Path(__file__).resolve().parent / f"kan_{problem_name.lower()}_{dim}d.pth"
            demo.torch.save({
                'models': [serialize_kan_model(model) for model in models],
                'x_data': x_data,
                'y_data': y_data,
                'problem_name': problem_name,
                'dim': dim,
            }, save_path)
            print(f"Saved KAN surrogate for {problem_name}-{dim}D to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain KAN surrogates for ZDT1-3 and DTLZ2-7 and save to disk")
    parser.add_argument("--kan_steps", type=int, default=25, help="Number of KAN training steps")
    parser.add_argument("--kan_hidden", type=int, default=10, help="KAN hidden width")
    parser.add_argument("--kan_grid", type=int, default=5, help="KAN grid size")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()
    pretrain_and_save_kan_surrogates(args)


if __name__ == "__main__":
    main()
