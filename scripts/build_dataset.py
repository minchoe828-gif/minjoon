import os 
import pandas as pd 
import quilt3
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def build_cell_dataset(target_structure='Control - DNA', num_samples=50, test_size=0.2):
    data_dir = './data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    csv_path = os.path.join(data_dir, 'metadata.csv')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' no file. first make metadata.csv by quilt3")

    df = pd.read_csv(csv_path, low_memory=False)

    df_filtered = df[df['StructureDisplayName'] == target_structure]

    if len(df_filtered) < num_samples:
        print(f'요청한 샘플 수 보다 데이터 부족, {len(df_filtered)}개 가져옴')
        sampled_df = df_filtered
    else:
        sampled_df = df_filtered.sample(n=num_samples, random_state=42)

    train_df, val_df = train_test_split(sampled_df, test_size=test_size, random_state=42)
    print(f'target data setup success: Train: {len(train_df)} Val: {len(val_df)}')

    pkg=quilt3.Package.browse('aics/pipeline_integrated_single_cell', 's3://allencell')

    def fetch_data(dataframe, target_folder):
        for row in tqdm(dataframe.itertuples(), total=len(dataframe)):
            target_s3_path = row.save_reg_path

            file_name = os.path.basename(target_s3_path)
            local_save_path = os.path.join(target_folder, file_name)

            if os.path.exists(local_save_path):
                continue
            try:
                pkg[target_s3_path].fetch(local_save_path)
            except Exception as e:
                print('download fail')

    fetch_data(train_df, train_dir)
    fetch_data(val_df, val_dir)

if __name__ == '__main__':
    build_cell_dataset(target_structure='Control - DNA', num_samples=20, test_size=0.2)



