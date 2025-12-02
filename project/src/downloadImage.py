import kagglehub
import shutil
import os

TARGET_DIR = "../data"


print("⬇ Downloading CIFAKE dataset...")
kaggle_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print("✔ KaggleHub dataset downloaded to:", kaggle_path)


if not os.path.exists(TARGET_DIR):
    print("📁 Copying dataset into ../data ...")
    shutil.copytree(kaggle_path, TARGET_DIR)
    print("✔ Dataset copied to:", TARGET_DIR)
else:
    print("✔ '../data' already exists — skipping copy.")
