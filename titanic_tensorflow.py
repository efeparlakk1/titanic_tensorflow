#import libs
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading data
data_ = sns.load_dataset("titanic")

#Getting info
data_.head()
data_.info()
data_.isnull().sum()
data_.describe().T

#function for preprocess
def preprocess(data):
    data.columns = [col.capitalize() for col in data.columns]
    data.drop("Deck", axis=1, inplace=True)
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    data['Pclass'] = data['Pclass'].astype('category')
    imputer = KNNImputer(n_neighbors=5)
    data['Age'] = imputer.fit_transform(data[['Age']]).astype("int")
    data.dropna(axis=0, inplace=True)
    return data

df = preprocess(data_)

def feature_engineering(data):
    num_cols = [col for col in data.columns if (data[col].dtypes in ["float64", "int64"]) & ("Survived" != col)]
    corr = data[num_cols].corrwith(data["Survived"]).reset_index()
    del_cols = corr.loc[(corr[0] > 0.8) | (corr[0] < -0.8), "index"]
    data = data.loc[:, ~data.columns.isin(del_cols)]
    data = pd.get_dummies(data, drop_first=True)
    scaler = MinMaxScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data

df_new = feature_engineering(df)

#train - test split
train_len = int(df_new.shape[0] * 0.85)
test_len = df_new.shape[0] - train_len
train_df = df_new[:train_len]
test_df = df_new[train_len:]

# sep X - y
X_train = tf.convert_to_tensor(train_df.drop(['Survived'], axis=1).values, dtype=tf.float32)
y_train = tf.convert_to_tensor(train_df['Survived'].values, dtype=tf.float32)

X_test = tf.convert_to_tensor(test_df.drop(['Survived'], axis=1).values, dtype=tf.float32)
y_test = tf.convert_to_tensor(test_df['Survived'].values, dtype=tf.float32)

#building model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

# Training model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

#plotting history (val - train losses)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



