import pandas as pd
import numpy as np
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

def model_creation(path: str, num_iter: int, test_size: float):
    df = data_prep(path)
    x = df.drop(columns=['y_yes'])
    y = df['y_yes']
########################### TASK 2 ##################################################################################
    # using sklearns train/test split, we can set stratified random sample with some desired split percentage (in this case 70%)
    # since assignment wants different number of iterations for mean and std dev of the model, run num_iter times 
    accuracy_scores = []
    f1_scores = []
    print('############################## test size: ', test_size, '############################################################')
    for _ in range(num_iter):
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,stratify=y) # y signifies we're stratifying on the target.

        model = GaussianNB()
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test,average='weighted')
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        print(f"The accuracy of my Naive Bayes Model for iteration {_} is:", accuracy)
        print(f"The F1 Score of my Naive Bayes Model for iteration {_} is:", f1)

    avg_acc = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    std_acc = np.std(accuracy_scores)
    std_f1 = np.std(f1_scores)

    print("avg accuracy:", avg_acc, ", std dev:", std_acc, ", avg f1:", avg_f1, ", std dev:", std_f1)
path = './data/bank-full.csv'
model_creation(path, 10, 0.3) # 70/30 split
########################### TASK 3 ##################################################################################

model_creation(path, 10, 0.9) # 10/90 split
model_creation(path, 10, 0.8) # 20/80 split
model_creation(path, 10, 0.7)
model_creation(path, 10, 0.6)
model_creation(path, 10, 0.5)
model_creation(path, 10, 0.4)
model_creation(path, 10, 0.3)
model_creation(path, 10, 0.2)
model_creation(path, 10, 0.1)
