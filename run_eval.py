import glob
import subprocess
import time
import os

from tqdm import tqdm


def kill_carla():
    print("Killing Carla server\n")
    time.sleep(1)
    subprocess.run(["killall", "-9", "CarlaUE4-Linux-Shipping"])
    time.sleep(4)


if __name__ == '__main__':
    selected_models = [f"model_{i}_steps.zip" for i in range(100000, 1100000, 100000)]
    tensorboard_path = './tensorboard'
    for model_path in tqdm(os.listdir(tensorboard_path), desc="Processing models"):
        config = model_path.split('id')[-1]
        print(config)
        print("==" * 30)
        print(model_path, config)
        model_ckpts = glob.glob(os.path.join(tensorboard_path, model_path, "*.zip"))
        model_ckpt_filenames = [os.path.basename(path) for path in model_ckpts]
        contains_all = all(model in model_ckpt_filenames for model in selected_models)
        if not contains_all:
            print("Skipping model {}, because not fully trained...".format(model_path))
            continue

        for model_ckpt in model_ckpts:
            if model_ckpt.split('/')[-1] not in selected_models: continue
            # summary_path = os.path.join(tensorboard_path, model_path, "eval",
            #                             os.path.basename(model_ckpt).replace(".zip", "_eval_summary.csv"))
            # if os.path.exists(summary_path):
            #     print("Already exists: ", summary_path)
            #     continue
            kill_carla()
            print(model_ckpt)

            args_eval = [
                "--model", model_ckpt,
                "--config", config,
                "--town", "Town02",  # default "Town02", change this to evaluate on other towns
                "--density", "regular",  # default "regular", change this to `dense` or `empty` to evaluate on other densities
            ]

            subprocess.run(["python", "eval.py"] + args_eval)
