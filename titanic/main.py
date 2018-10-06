# データの前処理につかうライブラリ
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

# 学習につかうライブラリ
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, StandardScaler
from sklearn import linear_model, tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, LeakyReLU

# 訓練データ取り込み
df_train = pd.read_csv('train.csv')
# テストデータ取り込み
df_test  = pd.read_csv('test.csv')
print(df_train.shape)

# Sex
lb_enc_sex = LabelEncoder() 
lb_sex = lb_enc_sex.fit_transform(df_train['Sex'])
oh_enc_sex = OneHotEncoder()
oh_enc_sex.fit(np.array(lb_sex).reshape(-1,1))

# Age
imp_age = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
imp_age.fit(np.array(df_train['Age']).reshape(-1, 1))

# Fare
imp_fare = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
imp_fare.fit(np.array(df_train['Fare']).reshape(-1, 1))

# Embarked
df_train['Embarked'] = df_train['Embarked'].fillna('U')
lb_enc_emb = LabelEncoder()
lb_emb = lb_enc_emb.fit_transform(df_train['Embarked'])
oh_enc_emb = OneHotEncoder()
oh_enc_emb.fit(np.array(lb_emb).reshape(-1,1))

def transform_data(df):

    # Sex
    lb_sex = lb_enc_sex.transform(df['Sex'])
    enc_sex = oh_enc_sex.transform(np.array(lb_sex).reshape(-1,1))
    df_sex = DataFrame(enc_sex.toarray(), columns=['male', 'female'])

    # Age
    age = imp_age.transform(np.array(df['Age']).reshape(-1, 1))
    df_age = DataFrame(age, columns=['Age'])

    # Fare
    fare = imp_fare.transform(np.array(df['Fare']).reshape(-1, 1))
    df_fare = DataFrame(fare, columns=['Fare'])

    # Embarked
    lb_emb = lb_enc_emb.transform(df['Embarked'])
    enc_emb = oh_enc_emb.transform(np.array(lb_emb).reshape(-1,1))
    df_emb = DataFrame(enc_emb.toarray(), columns=['C', 'Q', 'S', 'U'])   

    return pd.concat([df['Pclass'], df_sex, df['SibSp'], df['Parch'], df_fare, df_age, df_emb],axis=1)


# 学習に使用するデータを整形する
df_X = transform_data(df_train)
df_y = df_train['Survived']
print(df_X.shape)
# Feature Scaling
# 欠損値の削除
df_X = df_X.dropna()
sc = StandardScaler()
df_X = sc.fit_transform(df_X)

# 一応テストデータも整形しておく
df_X_test = transform_data(df_test)
df_X_test = sc.transform(df_X_test)

print(df_X.shape)
"""
inputs = Input(shape=(11,))
x = Dense(32)(inputs)
"""
file = open('data.txt','w')
file.writelines(str(df_X))