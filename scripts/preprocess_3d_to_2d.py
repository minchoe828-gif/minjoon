import pandas as pd
import numpy as np
from aicsimageio import AICSImage
from pathlib import Path
from tqdm import tqdm

def build_dataset_and_metadata(raw_dir, processed_dir, input_ch, target_ch, min_pixels=100):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata=[]

    for tif_path in tqdm(list(raw_dir.glob('*.tif*'))):
        reader = AICSImage(tif_path)
        target_3d = reader.get_image_data('ZYX', T=0, C=target_ch)
        input_3d = reader.get_image_data('ZYX', T=0, C=input_ch)

        for z_idx in range(target_3d.shape[0]):
            if np.sum(target_3d[z_idx] >0) >= min_pixels:
                file_prefix = f'{tif_path.stem}_z{z_idx:03d}'
                input_save_path = processed_dir / f"{file_prefix}_input.npy"
                target_save_path = processed_dir / f"{file_prefix}_target.npy"

                np.save(input_save_path, input_3d[z_idx])
                np.save(target_save_path, input_3d[z_idx])

                metadata.append({
                    'original_file': tif_path.name,
                    'z_index': z_idx,
                    'input_path': str(input_save_path.name),
                    'target_path': str(target_save_path.name)
                })
    
    df = pd.DataFrame(metadata)
    df.to_csv(processed_dir / 'valid_slices_metadata.csv', index=False)
    print(f'Total valid 2D slices extracted: {len(df)}')