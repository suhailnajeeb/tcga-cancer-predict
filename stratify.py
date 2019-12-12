import h5py
from sklearn.model_selection import train_test_split

dbPath = 'C:\Data\data.h5'
db = h5py.File(dbPath, 'r')

nTotal = db["RNASeq"].shape[0]
nFeat = db["RNASeq"].shape[1]

X = list(range(nTotal))
y = db["label"][...]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)