import pandas as pd
def data_prep(path: str):
    df = pd.read_csv(path, delimiter=';')
    encoded_df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y'], dtype='int64', drop_first=True)
    # testing performance with dropping unknown values along with nan
    df_drop_unknown = encoded_df.drop(columns=[col for col in df.columns if col.endswith('_unknown')])
    return df_drop_unknown.dropna()

df = data_prep('./data/bank-full.csv')
df.to_csv('./data/cleaned_data.csv')
