from pathlib import Path

import db_zdt1_eva as base


base.SOURCE_PROBLEMS = ["ZDT1"]
base.MODEL_PATH = "db_saea_zdt1_self_only.pth"
base.TRAIN_LOG_PATH = base.REWARD_LOG_DIR / "db_zdt1_demo_train_rewards.json"
base.INFER_RESULT_PATH = Path(__file__).resolve().parent / "db_zdt1_demo_infer_results.json"
base.INFER_NOTEBOOK_PATH = Path(__file__).resolve().parent / "db_zdt1_demo_plots.ipynb"


if __name__ == "__main__":
    base.main()
