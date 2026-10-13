import pandas as pd
import numpy as np
# import performance metric libraries
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
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
    return avg_acc,std_acc,avg_f1,std_f1
    
path = './data/bank-full.csv'
model_creation(path, 10, 0.3) # 70/30 split
########################### TASK 3 ##################################################################################
data = {}
avg_accs_list = []
std_accs_list = []
avg_f1_list = []
std_f1_list = []
iter_count = []
for i in range(9):
    avg_acc, std_acc, avg_f1, std_f1 = model_creation(path, 10, ((9-i)*10)/100) # from 90% test to 10% test
    avg_accs_list.append(avg_acc)
    std_accs_list.append(std_acc)
    avg_f1_list.append(avg_f1)
    std_f1_list.append(std_f1)
    iter_count.append(i+1)

data['iteration'] = iter_count
data['average accuracies'] = avg_accs_list
data['accuracy stds'] = std_accs_list
data['average f1 scores'] = avg_f1_list
data['f1 stds'] = std_f1_list

fig, axs = plt.subplots(2,2,figsize=(10,4))

axs[0,0].plot(data['iteration'], data['average accuracies'])
axs[0,0].set_title('average accuracy as training data increases')
axs[0,0].set_xlabel('10$x$% set for training')
axs[0,0].set_ylabel('accuracy')

axs[0,1].plot(data['iteration'],data['accuracy stds'])
axs[0,1].set_title('standard deviations of accuracies')
axs[0,1].set_xlabel('10$x$% set for training')
axs[0,1].set_ylabel('accuracy standard deviation')

axs[1,0].plot(data['iteration'], data['average f1 scores'])
axs[1,0].set_title('average f1 scores as training data increases')
axs[1,0].set_xlabel('10$x$% set for training')
axs[1,0].set_ylabel('average f1 scores')

axs[1,1].plot(data['iteration'],data['f1 stds'])
axs[1,1].set_title('standard deviations of accuracies')
axs[1,1].set_xlabel('10$x$% set for training')
axs[1,1].set_ylabel('f1 standard deviation')

plt.tight_layout()
plt.show()

df = pd.DataFrame(data)
print(df.to_latex(index=False, float_format="{:.4f}".format))
