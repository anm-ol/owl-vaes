import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

class T3Dataset(IterableDataset):
    def __init__(self, root="t3_data/"):
        super().__init__()
        self.root = root
        
        # This will store lightweight pointers: (path_to_npz_file, frame_index)
        self.valid_frame_pointers = []
        
        print("Indexing valid frames (this runs only once)...")
        if os.path.isdir(root):
            npz_files = [f for f in os.listdir(root) if f.endswith(".npz")]
            for npz_file in npz_files:
                npz_path = os.path.join(root, npz_file)
                try:
                    # Use memory mapping to quickly read just the mask without loading all images
                    with np.load(npz_path, mmap_mode='r') as data:
                        # Ensure you are using the correct key for your mask
                        mask = data.get('valid_frames')
                        if mask is None:
                            print(f"Warning: No 'valid_frames' key in {npz_file}, skipping.")
                            continue

                        # Find all indices where the mask is 1
                        valid_indices = np.where(mask == 1)[0]
                    
                    # Create a pointer for each valid frame
                    for idx in valid_indices:
                        self.valid_frame_pointers.append((npz_path, idx))
                
                except Exception as e:
                    print(f"Warning: Could not index file {npz_file}. Error: {e}")
                    continue
        
        if not self.valid_frame_pointers:
            raise FileNotFoundError("No valid frames could be indexed from the dataset directory.")
            
        print(f"Found {len(self.valid_frame_pointers)} total valid frames.")

    def __iter__(self):
        # Shuffle the list of all valid frames at the beginning of each epoch
        random.shuffle(self.valid_frame_pointers)
        
        worker_info = get_worker_info()
        # For multi-process data loading, each worker gets a unique slice of the data
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            pointers_to_process = self.valid_frame_pointers[worker_id::num_workers]
        else:
            pointers_to_process = self.valid_frame_pointers

        for file_path, frame_idx in pointers_to_process:
            try:
                # Load the specific frame from disk only when needed
                with np.load(file_path, mmap_mode='r') as data:
                    frame = data['images'][frame_idx]
                
                yield torch.from_numpy(frame).float()

            except Exception as e:
                print(f"Warning: Skipping frame {frame_idx} from {os.path.basename(file_path)}. Error: {e}")
                continue

def collate_fn(frames):
    batch = torch.stack(frames)
    
    # Robust normalization that works for both uint8 [0,255] and float [0,1] data
    if batch.max() > 1.0:
        batch = batch / 255.0
    
    batch = batch * 2.0 - 1.0
    return batch

def get_loader(batch_size, **data_kwargs):
    """
    Creates an optimized DataLoader.
    """
    dataset = T3Dataset(**data_kwargs)
    
    # Using multiple workers is critical for performance with lazy loading
    num_workers = min(os.cpu_count(), 8)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True, # Helps speed up CPU to GPU data transfer
        persistent_workers=True # Avoids re-initializing workers every epoch
    )
    return loader