import pandas as pd


def preprocess_df(df):
    df = df[df['Gender'] != 2.0]
    df = df[df['Age'] != 6.0]
    df = df[df['Type'] != 'log']

    return df

df_train = pd.read_csv('train_df.csv')
df_val = pd.read_csv('val_df.csv')

df_train = preprocess_df(df_train)
df_val = preprocess_df(df_val)

df = pd.concat((df_train, df_val))
print(len(df), len(df_train), len(df_val))

def analyze_data(df):
    df = df[df['Type'] != 'log']
    df = df[df['Gender'] != 2.0]
    print(len(df))

    count = df.groupby('Age').count()['Path']
    percent = count / len(df) * 100
    analyze_df = pd.concat((count, percent), axis=1)

    print(analyze_df)


analyze_data(df_val)