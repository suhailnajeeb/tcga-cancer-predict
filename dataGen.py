import pandas as pd
import h5py
import numpy as np
import progressbar

data = 'C:\Data\HiSeqV2'
labels = 'C:\Data\TCGA_phenotype_denseDataOnlyDownload.tsv'
dbPath = 'C:\Data\data.h5'
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

nTotal = df.shape(0)    #10459
nFeat = df.shape(1)     #20530

print('Total Number of samples: '+ str(nTotal))
print('Features (RNASeq) per sample: ' + str(nFeat))

print('Diseases to predict: ')

diseases = labeldf._primary_disease.unique()

for disease in diseases:
    print(disease)

# Defining Categorical values for each disease

diseasedict = {
    'skin cutaneous melanoma':0, 'thyroid carcinoma':1, 'sarcoma':2,
    'prostate adenocarcinoma':3, 'pheochromocytoma & paraganglioma':4,
    'pancreatic adenocarcinoma':5, 'head & neck squamous cell carcinoma':6,
    'esophageal carcinoma':7, 'colon adenocarcinoma':8,
    'cervical & endocervical cancer':9, 'breast invasive carcinoma':10,
    'bladder urothelial carcinoma':11, 'testicular germ cell tumor':12,
    'kidney papillary cell carcinoma':13, 'kidney clear cell carcinoma':14,
    'acute myeloid leukemia':15, 'rectum adenocarcinoma':16,
    'ovarian serous cystadenocarcinoma':17, 'lung adenocarcinoma':18,
    'liver hepatocellular carcinoma':19,
    'uterine corpus endometrioid carcinoma':20, 'glioblastoma multiforme':21,
    'brain lower grade glioma':22, 'uterine carcinosarcoma':23, 'thymoma':24,
    'stomach adenocarcinoma':25, 'diffuse large B-cell lymphoma':26,
    'lung squamous cell carcinoma':27, 'mesothelioma':28,
    'kidney chromophobe':29, 'uveal melanoma':30, 'cholangiocarcinoma':31,
    'adrenocortical cancer':32
}

print('Creating Database File at :' + dbPath)
db = h5py.File(dbPath, mode = 'w')

print('Setting up Database')
db.create_dataset("name", (nTotal,), np.dtype('|S16'))
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
        db["RNASeq"][idx] = np.asarray(row, dtype = np.float32)
        db["label"][idx] = np.uint8(diseasedict[data[2]])
        idx = idx + 1
    except:
        print("Error: Cannot find label")
        continue

db.close()