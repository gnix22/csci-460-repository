import pandas as pd
# import performance metric libraries
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
# data preprocessing method
#df = pd.read_csv('./data/bank-full.csv',delimiter=';')
def data_prep(path: str):
    df = pd.read_csv(path, delimiter=';')
    # removing these columns, need to check
    encoded_df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y'], dtype='int64', drop_first=True)
    return encoded_df.dropna()

df = data_prep('./data/bank-full.csv')
print(df.columns)

