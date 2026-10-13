import pandas as pd
# import performance metric libraries
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
########################### TASK 1 ##################################################################################
# data preprocessing method
def data_prep(path: str):
    df = pd.read_csv(path, delimiter=';')
    encoded_df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y'], dtype='int64', drop_first=True)
    return encoded_df.dropna()

df = data_prep('./data/bank-full.csv')
x = df.drop(columns=['y_yes'])
y = df['y_yes']
########################### TASK 2 ##################################################################################
# using sklearns train/test split, we can set stratified random sample with some desired split percentage (in this case 70%)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y) # y signifies we're stratifying on the target.

model = GaussianNB()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(pred)
