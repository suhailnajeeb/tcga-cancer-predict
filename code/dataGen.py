import pandas as pd
import h5py
import numpy as np
import progressbar
from labelMapping import diseasedict

data = 'C:/Data/HiSeqV2'
labels = 'C:/Data/TCGA_phenotype_denseDataOnlyDownload.tsv'
dbPath = 'E:/Data/HiSeqV2.h5'
inputDim2D = True
inputDim = (116, 177)       # nFeat = 20351 = padded bottom = 20532 = 116 x 177
verbose = False

print('Loading data ... Patience.')
df = pd.read_csv(data, sep='\t').transpose()

print('Loading labels ...')
labeldf = pd.read_csv(labels, sep = '\t')

print('Housekeeping ...')
df.columns = df.iloc[0]
df = df.drop('Sample', axis = 0)

labeldf = labeldf.set_index('sample')

# dimensions: 10459 x 20530

nTotal = df.shape[0]    #10459
nFeat = df.shape[1]     #20530

print('Total Number of samples: '+ str(nTotal))
print('Features (RNASeq) per sample: ' + str(nFeat))

print('Diseases to predict: ')

diseases = labeldf._primary_disease.unique()

for disease in diseases:
    print(disease)

print('Creating Database File at : ' + dbPath)
db = h5py.File(dbPath, mode = 'w')

print('Setting up Database')
db.create_dataset("name", (nTotal,), np.dtype('|S16'))
if inputDim2D:
    db.create_dataset("RNASeq", (nTotal,) + inputDim, np.float32)
else:
    db.create_dataset("RNASeq", (nTotal, nFeat), np.float32)
db.create_dataset("label", (nTotal,), np.uint8)

idx = 0

print('Writing ' + str(nTotal) + ' samples to Dataset')

for index,row in progressbar.progressbar(df.iterrows(), redirect_stdout=True):
    try:
        data = labeldf.loc[index]
        if(verbose):
            print('Processing '+ str(idx) + ' of ' + str(nTotal) + ' : ' + index + '\t disease: \t' + str(data[2]))
        db["name"][idx] = np.asarray(index, dtype = np.dtype('|S16'))
        if inputDim2D:
            row = np.asarray(row, dtype=np.float32)
            row = np.append(row, np.array([0.0, 0.0])) # 0 padding at the end to make length 20532
            db["RNASeq"][idx] = np.reshape(row, inputDim)
        else:
            db["RNASeq"][idx] = np.asarray(row, dtype = np.float32)

        db["label"][idx] = np.uint8(diseasedict[data[2]])
        idx = idx + 1
    except Exception as err:
        print("Error: ", err)
        # print("Error: Cannot find label")
        continue

print('Closing Database ..')
db.close()
print('Complete!')
