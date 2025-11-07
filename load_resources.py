import yadisk
import zipfile
from tqdm import tqdm

DATASET_URL = "https://disk.360.yandex.ru/d/9k_k6G6a03GURg"

y = yadisk.YaDisk() 
meta = y.get_public_meta(DATASET_URL)
total_size = meta.size
file_name = meta.name
print(f"Downloading {file_name} ({total_size / 1e6:.2f} MB)")

y.download_public(DATASET_URL, file_name)

'''
with y.download_by_link(link, file_name, stream=True) as resp:
    print("Got response from disk")
    resp.raise_for_status()
    chunk_size = 8192
    with open(file_name, "wb") as f, tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk is None:
                   continue
            f.write(chunk)
            pbar.update(len(chunk))
'''

with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall(".")