import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import torchvision.transforms.functional as TF

class T3LiveMergeDataset(IterableDataset):
    def __init__(self, root="t3_data/", pose_root="t3_pose/", pose_suffix="_two_player_poses.npz", target_size=(256, 256)):
        super().__init__()
        self.root = root
        self.pose_root = pose_root
        self.pose_suffix = pose_suffix
        self.target_size = tuple(target_size)
        
        # This will store tuples of (path_to_original_npz, path_to_pose_npz, frame_index)
        self.valid_frame_pointers = []
        
        print("Indexing valid frames using 'attention_mask'...")
        if os.path.isdir(self.root):
            npz_files = [f for f in os.listdir(self.root) if f.endswith(".npz")]
            for npz_file in npz_files:
                original_path = os.path.join(self.root, npz_file)
                pose_path = os.path.join(self.pose_root, npz_file.replace('.npz', self.pose_suffix))
                
                if not os.path.exists(pose_path):
                    continue
                
                try:
                    # Load just the attention mask to find the valid range
                    with np.load(original_path, mmap_mode='r') as data:
                        mask = data['valid_frames']
                        # Find the index of the last valid frame
                        last_valid_idx = int(np.where(mask == 1)[0][-1])
                    
                    # Create pointers for all frames up to the last valid one
                    for idx in range(last_valid_idx + 1):
                        self.valid_frame_pointers.append((original_path, pose_path, idx))
                
                except Exception as e:
                    print(f"Warning: Could not index file {npz_file}. Error: {e}")
                    continue
        
        if not self.valid_frame_pointers:
            raise FileNotFoundError("No valid frames found in the dataset directories.")
            
        print(f"Found {len(self.valid_frame_pointers)} total valid frame pointers.")

    def __iter__(self):
        # Step 1: Divide the list of pointers among workers
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            all_pointers = self.valid_frame_pointers
            start = worker_id * len(all_pointers) // num_workers
            end = (worker_id + 1) * len(all_pointers) // num_workers
            worker_pointers = all_pointers[start:end]
        else:
            worker_pointers = self.valid_frame_pointers
        
        # Step 2: Group pointers by file path to reduce file I/O
        # This will store {path: [frame_idx1, frame_idx2, ...]}
        grouped_pointers = {}
        for orig_path, pose_path, frame_idx in worker_pointers:
            if orig_path not in grouped_pointers:
                grouped_pointers[orig_path] = {'pose_path': pose_path, 'indices': []}
            grouped_pointers[orig_path]['indices'].append(frame_idx)

        # Shuffle the files to maintain randomness
        file_paths = list(grouped_pointers.keys())
        random.shuffle(file_paths)
        
        # Step 3: Iterate through files, open once, and process all frames
        for orig_path in file_paths:
            pose_path = grouped_pointers[orig_path]['pose_path']
            indices = grouped_pointers[orig_path]['indices']
            
            try:
                # Open the files once per file
                with np.load(orig_path, mmap_mode='r') as original_data, \
                     np.load(pose_path, mmap_mode='r') as pose_data:
                    
                    images = original_data['images']
                    pose_images = pose_data['pose_images']

                    # Iterate through the frames for this specific file
                    for frame_idx in indices:
                        orig_frame = torch.from_numpy(images[frame_idx]).float()
                        pose_frame_raw = pose_images[frame_idx]
                        pose_frame = torch.from_numpy(pose_frame_raw).float()
                        
                        if pose_frame.ndim == 3 and pose_frame.shape[0] == 3:
                            pose_frame = torch.max(pose_frame, dim=0, keepdim=True)[0]

                        pose_frame = TF.gaussian_blur(pose_frame.unsqueeze(0), kernel_size=3).squeeze(0)
                        
                        combined_frame = torch.cat([orig_frame, pose_frame], dim=0)
                        resized_frame = F.interpolate(combined_frame.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
                        
                        yield resized_frame

            except Exception as e:
                print(f"Warning: Skipping file {os.path.basename(orig_path)}. Error: {e}")
                continue

def collate_fn(frames):
    batch = torch.stack(frames)
    
    # Robust normalization: check if data is [0, 255] or [0, 1]
    if batch.max() > 1.0:
        batch = batch / 255.0
    
    # Scale from [0, 1] to [-1, 1]
    batch = batch * 2.0 - 1.0
    return batch

# In the get_loader function
def get_loader(batch_size, **data_kwargs):
    dataset = T3LiveMergeDataset(**data_kwargs)
    
    # Use more workers to match the 12 CPUs you allocated per task in your job script.
    num_workers = 12 
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        # This can sometimes help with worker initialization speed
        persistent_workers=True 
    )
    return loader
# ----------------- Main Execution Block (for direct testing) -----------------
# This block is for testing the loader independently and is not used by train.py
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Simple setup for running this file directly
    sys.path.append(str(Path(__file__).parent.parent))

    # Assuming a folder structure for testing
    TEST_ROOT = "t3_data/"
    TEST_POSE_ROOT = "t3_pose/"
#changed batch size to 16 from 4
    try:
        loader = get_loader(batch_size=4, root=TEST_ROOT, pose_root=TEST_POSE_ROOT, target_size=(256, 256))
        print("✅ Data loader initialized. Testing one batch.")
        
        # Use a single, non-overlapping progress bar for clean output
        for batch in tqdm(loader, total=1):
            print(f"Batch shape: {batch.shape}")
            print(f"Batch values range: [{batch.min():.2f}, {batch.max():.2f}]")
            break

    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Please check the paths in the script.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")