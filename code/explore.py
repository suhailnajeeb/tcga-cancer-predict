import pandas as pd
import h5py
import numpy as np
from labelMapping import diseasedict

data_path = 'C:/Data/HiSeqV2'
label_path = 'C:/Data/TCGA_phenotype_denseDataOnlyDownload.tsv'
db_path = 'E:/Data/HiSeqV2_1D.h5'

df = pd.read_csv(data_path, sep = '\t')
df = df.transpose()
df.columns = df.iloc[0]
df = df.drop('Sample', axis = 0)

label_df = pd.read_csv(label_path, sep = '\t')
label_df = label_df.set_index('sample')

n_total = df.shape[0]
n_feat = df.shape[1]

# Create 1D Dataset:

db = h5py.File(db_path, mode = 'w')
db.create_dataset('name', (n_total,), np.dtype('|S16'))
db.create_dataset('RNASeq', (n_total, n_feat), np.float32)
db.create_dataset('label', (n_total,), np.uint8)

idx = 0

for index, row in df.iterrows():
    try:
        data = label_df.loc[index]
        db['RNASeq'][idx] = np.uint8(diseasedict[data[2]])
        idx = idx + 1
    except Exception as err:
        print("Error: " + err)
        continue

db.close()