from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal
import pandas as pd
import quilt3
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from loguru import logger
from src.utils.logger_setup import setup_project_logger

setup_project_logger()

@logger.catch(reraise=False)
#s3_path의 cell image를 local_path에 저장시키는 함수이다.
def download_single_file(pkg: quilt3.Package, s3_path: str, local_path: Path, retries: int = 3) -> bool:
    if local_path.exists():
        return True

    for attempt in range(retries):
        try:
            pkg[s3_path].fetch(str(local_path))
            return True
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to fetch {s3_path} after {retries} attempts: {e}")
                return False
            logger.warning(f"Fetch failed for {s3_path}, retrying ({attempt + 1}/{retries})...")
            
    return False

@logger.catch
#df를 바탕으로 주어진 폴더에 image를 병렬적으로 다운받는 함수이다. 
def fetch_data_concurrently(df: pd.DataFrame, target_folder: Path, pkg: quilt3.Package, max_workers: int = 8) -> None:
    if 'save_reg_path' not in df.columns:
        raise ValueError("The metadata CSV does not contain the required 'save_reg_path' column.")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {}
        for row in df.itertuples():
            s3_path = str(row.save_reg_path)
            if pd.isna(s3_path):
                continue
            
            local_path = target_folder / Path(s3_path).name
            future = executor.submit(download_single_file, pkg, s3_path, local_path)
            future_to_path[future] = s3_path

        with tqdm(total=len(future_to_path), desc=f"Downloading to {target_folder.name}") as pbar:
            for future in as_completed(future_to_path):
                s3_path = future_to_path[future]
                try:
                    is_success = future.result()
                    if is_success:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"Unhandled exception occurred for {s3_path}: {e}")
                    fail_count += 1
                finally:
                    pbar.update(1)

    logger.info(f"[{target_folder.name}] Fetch complete. Success: {success_count}, Failed: {fail_count}")

@logger.catch
#dataset을 위한 cell 이미지들을 다운받아서 각각 data/raw/images/train, data/raw/images/val 에 저장시키는 함수이다. 
def build_cell_dataset(
    num_samples: int | Literal['all'] = 1, 
    test_size: float = 0.2, 
    data_dir: str | Path = './data/raw',
    max_workers: int = 8
) -> None:

    base_dir = Path(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = base_dir / 'images' / 'train'
    val_dir = base_dir / 'images' / 'val'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to Allen Institute Quilt3 registry...")
    pkg = quilt3.Package.browse('aics/pipeline_integrated_single_cell', 's3://allencell')
    
    target_file = "metadata.csv"
    
    if target_file not in pkg:
        raise FileNotFoundError(
            f"Metadata file '{target_file}' missing in the Quilt3 package 'aics/pipeline_integrated_single_cell'."
            "Please check the exact file name in the package contents."
        )
    
    df_meta = pkg[target_file]()
    
    if isinstance(num_samples, int):
        actual_n = min(num_samples, len(df))
        sampled_df = df_meta.sample(n=actual_n, random_state=42)
        if actual_n < num_samples:
            logger.warning(f"Requested {num_samples} samples, but only {actual_n} available. Using all data")
        
    elif num_samples == 'all':
        sampled_df = df_meta
    else:
        raise ValueError(f"num_samples argument must be an integer or 'all'") 

    train_df, val_df = train_test_split(sampled_df, test_size=test_size, random_state=42)
    logger.info(f"Target data split complete | Train: {len(train_df)} | Val: {len(val_df)}")

    logger.info("Connecting to Allen Institute Quilt3 registry...")
    pkg = quilt3.Package.browse('aics/pipeline_integrated_single_cell', 's3://allencell')

    fetch_data_concurrently(train_df, train_dir, pkg, max_workers=max_workers)
    fetch_data_concurrently(val_df, val_dir, pkg, max_workers=max_workers)

if __name__ == '__main__':
    build_cell_dataset(num_samples=20, test_size=0.2)
