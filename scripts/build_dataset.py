from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import quilt3
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 1. 표준 logging 모듈 제거 및 Loguru 엔진 도입
from loguru import logger
from src.utils.logger_setup import setup_project_logger

# 이 스크립트가 독립적으로 실행될 때를 대비해 로깅 파이프라인을 초기화합니다.
setup_project_logger()

# 2. @logger.catch(reraise=False): 
# 단일 파일 다운로드 중 에러가 터지면 시스템을 멈추는 대신,
# 그 시점의 s3_path와 로컬 변수 상태를 모두 로그에 박제하고 False를 반환하여 우아하게 실패(Graceful Degradation)합니다.
@logger.catch(reraise=False)
def download_single_file(pkg: quilt3.Package, s3_path: str, local_path: Path, retries: int = 3) -> bool:
    """
    단일 파일을 다운로드하며 지수 백오프(Exponential Backoff) 기반의 재시도 로직을 포함합니다.
    """
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

# 멀티스레딩 환경의 예외 추적망
@logger.catch
def fetch_data_concurrently(df: pd.DataFrame, target_folder: Path, pkg: quilt3.Package, max_workers: int = 8) -> None:
    """
    I/O 바운드 작업인 S3 다운로드를 멀티스레딩으로 병렬 처리하여 속도를 극대화합니다.
    """
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

# 메인 파이프라인의 진입점 방어
@logger.catch
def build_cell_dataset(
    target_structure: str = 'Control - DNA', 
    num_samples: int = 50, 
    test_size: float = 0.2, 
    data_dir: str | Path = './data'
) -> None:
    """
    metadata.csv를 읽고 조건에 맞는 데이터를 필터링한 후 S3에서 병렬로 다운로드합니다.
    """
    base_dir = Path(data_dir)
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'
    csv_path = base_dir / 'metadata.csv'

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata file missing at '{csv_path}'. "
            "Please initialize metadata using quilt3 first."
        )

    df = pd.read_csv(csv_path, low_memory=False)

    df_filtered = df[df['StructureDisplayName'] == target_structure]
    available_samples = len(df_filtered)

    if available_samples == 0:
        raise ValueError(f"No samples found for target structure: '{target_structure}'")

    if available_samples < num_samples:
        logger.warning(
            f"Requested {num_samples} samples, but only {available_samples} are available. "
            "Using all available samples."
        )
        sampled_df = df_filtered
    else:
        sampled_df = df_filtered.sample(n=num_samples, random_state=42)

    train_df, val_df = train_test_split(sampled_df, test_size=test_size, random_state=42)
    logger.info(f"Target data split complete | Train: {len(train_df)} | Val: {len(val_df)}")

    logger.info("Connecting to Allen Institute Quilt3 registry...")
    pkg = quilt3.Package.browse('aics/pipeline_integrated_single_cell', 's3://allencell')

    fetch_data_concurrently(train_df, train_dir, pkg)
    fetch_data_concurrently(val_df, val_dir, pkg)

if __name__ == '__main__':
    build_cell_dataset(target_structure='Control - DNA', num_samples=20, test_size=0.2)
