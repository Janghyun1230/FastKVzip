import glob
import os
import shutil

for data in [
    # "needle",
    # "squad",
    # "gsm",
    # "scbench_many_shot",  # 26474
    # "scbench_mf",  # 149860
    # "scbench_choice_eng",  # 119299
    # "scbench_qa_eng",  # 122101
    # "scbench_repoqa",  # 72499
    # "scbench_kv",  # 169428
    # "scbench_prefix_suffix",  # 112635
    # "scbench_summary",  # 117806
    # "scbench_vt",  # 124551
    "scbench_kv_short",  # 169428
]:
    for idx in range(100):
        name = "qwen3-8b"
        name2 = "qwen3-8b"

        path = os.path.join(f"/root/code/kvzip/results_old/{data}/{idx}_{name}")
        path2 = os.path.join(f"/root/code/kvzip/results/{data}/{idx}_{name2}")
        if os.path.isdir(path):
            shutil.copytree(path, path2)
        # if os.path.isdir(path):
        #     os.rename(path, path2)


# for dirpath, dirnames, filenames in os.walk("./"):
#     for dirname in dirnames:
#         if "_half" in dirname:
#             new_dirname = "".join(dirname.split("_half"))
#             if new_dirname != dirname:
#                 old_path = os.path.join(dirpath, dirname)
#                 new_path = os.path.join(dirpath, new_dirname)
#                 print(f"Renaming:\n  {old_path}\n→ {new_path}")
#                 os.rename(old_path, new_path)
