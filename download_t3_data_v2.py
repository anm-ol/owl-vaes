
from huggingface_hub import login, hf_hub_download
import numpy as np
import os

# Authenticate with your token
login(token="")

def download_t3_data_v2(save_dir="./t3_data"):
    from huggingface_hub import list_repo_files
    os.makedirs(save_dir, exist_ok=True)
    # List all files in the dataset repo
    all_files = list_repo_files("Summer-193/t3_data_npz", repo_type="dataset")
    npz_files = [f for f in all_files if f.endswith(".npz")]
    print(f"Found {len(npz_files)} npz files tekken_dataset_npz/P1_WIN")
    if not npz_files:
        print("No .npz files found. Exiting.")
        return
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_npz(npz_file):
        try:
            print(f"Downloading {npz_file}...")
            file_path = hf_hub_download(
                repo_id="Summer-193/t3_data_npz",
                filename=npz_file,
                cache_dir=save_dir,
                repo_type="dataset"
            )
            dest_path = os.path.join(save_dir, os.path.basename(npz_file))
            shutil.copy(file_path, dest_path)
            print(f"Saved raw npz to: {dest_path}")
        except Exception as e:
            print(f"Error downloading {npz_file}: {e}")

    max_workers = min(8, len(npz_files))  # Use up to 8 threads or less if fewer files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_npz, npz_file) for npz_file in npz_files]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    save_dir = "./t3_data"
    download_t3_data_v2(save_dir)
