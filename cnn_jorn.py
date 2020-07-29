Top# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 08:01:30 2020

@author: jbeuk
"""

#%%
from pyts.image import GramianAngularField
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("C:/Users/jbeuk/Documents/Jorn/stock_cnn_blog_pub-master/stock_cnn_blog_pub-master/stock_history/WMT/WMT.csv")

def dataXY(df):
    def create_labels(series, window):
        prominence=3
        fpmax = find_peaks(series,prominence=prominence)[0]
        fpmin = find_peaks(-series,prominence=prominence)[0]
        y = np.array(series.shape[0]*[2])
        y = np.expand_dims(y,-1)
        y[fpmax,:] = 0
        y[fpmin,:] = 1
        y = y[window:,:]
        y = OneHotEncoder(sparse=False, categories='auto').fit_transform(y)
        return y

    def GAF(series,window,image_size):
        gasf = GramianAngularField(image_size=image_size, method='summation')
        j=0
        X_gasf = np.zeros((series.shape[1]-window,image_size,image_size,series.shape[0]))
        for i in range(0,series.shape[1]-window):
            X_gasf[j,:,:,:] = np.moveaxis(gasf.transform(series[:,i:i+window]), (0), (-1))
            j+=1
        return X_gasf

    train_size = 0.5
    cv_size = 0.3
    window = 50

    b = abs(df.iloc[:,[2,3]].diff(axis=1)).iloc[:,1:2].to_numpy()
    c = df.iloc[:,[4,6]].to_numpy()
    d = np.concatenate((b,c),1)

    e = d[0:round(df.shape[0]*train_size),:].T
    x_train = GAF(e,window,50)
    y_train = create_labels(df['close'][0:round(df.shape[0]*train_size)], window)

    e = d[round(df.shape[0]*train_size):round(df.shape[0]*(train_size+cv_size)),:].T
    x_cv = GAF(e,window,50)
    y_cv = create_labels(df['close'][0:round(df.shape[0]*train_size)], window)

    e = d[round(df.shape[0]*(train_size+cv_size)):,:].T
    x_test = GAF(e,window,50)
    y_test = create_labels(df['close'][0:round(df.shape[0]*train_size)], window)

    return x_train, y_train, x_cv, y_cv, x_test, y_test


dataXY(df)
# a = df.iloc[0:round(df.shape[0]*0.5),1:4].to_numpy().T
b = abs(df.iloc[:,[2,3]].diff(axis=1)).iloc[:,1:2].to_numpy()
c = df.iloc[:,[4,6]].to_numpy()
d = np.concatenate((b,c),1)
e = d[0:round(df.shape[0]*0.5),:].T

x_train = GAF(e,window,50)
y_train = create_labels(df['close'][0:round(df.shape[0]*0.5)], window)

# a = df.iloc[round(df.shape[0]*0.5):round(df.shape[0]*0.7),1:4].to_numpy().T
a = df.iloc[round(df.shape[0]*0.5):round(df.shape[0]*0.7),[2,4,6]].to_numpy().T
x_cv = GAF(a,window,50)
y_cv = create_labels(df['close'][round(df.shape[0]*0.5):round(df.shape[0]*0.7)], window)

# a = df.iloc[round(df.shape[0]*0.7):,1:4].to_numpy().T
a = df.iloc[round(df.shape[0]*0.7):,[2,4,6]].to_numpy().T
x_test = GAF(a,window,50)
x_test = np.moveaxis(x_test, (0,3), (3,0))
y_test = create_labels(df['close'][round(df.shape[0]*0.7):], window)


#%%
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import numpy as np



def dataXY(df):
    def create_labels(y):
        prominence=10
        fpmax = find_peaks(y,prominence=prominence)[0]
        fpmin = find_peaks(-y,prominence=prominence)[0]
        scaler = MinMaxScaler((0,1))
        labels = np.zeros((len(y)))
        for i in range(len(fpmax)):
            Dmax = fpmax[i]
            if np.where(fpmin>fpmax[i])[0].shape[0]>0:
                EVmin = fpmin[np.where(fpmin>fpmax[i])[0][0]]
                labels[Dmax:EVmin] = scaler.fit_transform((y.iloc[Dmax:EVmin].to_numpy()).reshape(-1,1))[:,0]
        for i in range(len(fpmin)):
            Dmin = fpmin[i]
            if np.where(fpmax>fpmin[i])[0].shape[0]>0:
                EVmax = fpmax[np.where(fpmax>fpmin[i])[0][0]]
                labels[Dmin:EVmax] = scaler.fit_transform((y.iloc[Dmin:EVmax].to_numpy()).reshape(-1,1))[:,0]
        return labels

    def create_X(x):
        x1 = add_all_ta_features(x,'open','high','low','close','volume',fillna=True)
        x1 = x1.iloc[:,7:]
        return x1

    def delZero(x,y):
        del_idx = np.where(y!=0)[0][0]
        y = y[del_idx:]
        x = x[del_idx:]
        del_idx = np.where(y!=0)[0][-1]
        y = y[:del_idx]
        x = x[:del_idx]
        return x,y

    train_size = 0.7
    cv_size = 0

    y = df['close']
    x = df.iloc[:,1:]

    x_train = create_X(x[0:round(y.shape[0]*train_size)])
    y_train = create_labels(y[0:round(y.shape[0]*train_size)])
    x_train, y_train = delZero(x_train,y_train)

    # x_cv = create_X(x[round(x.shape[0]*train_size):round(x.shape[0]*(train_size+cv_size))])
    # y_cv = create_labels(y[round(x.shape[0]*train_size):round(x.shape[0]*(train_size+cv_size))])

    x_test = create_X(x[round(x.shape[0]*(train_size+cv_size)):])
    y_test = create_labels(y[round(x.shape[0]*(train_size+cv_size)):])
    x_test, y_test = delZero(x_test,y_test)

    return x_train, y_train, x_test, y_test


df = pd.read_csv("C:/Users/jbeuk/Documents/Jorn/stock_cnn_blog_pub-master/stock_cnn_blog_pub-master/stock_history/WMT/WMT.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp',inplace=True)
df.reset_index(inplace=True,drop=True)
x_train, y_train, x_test, y_test = dataXY(df)




#%%
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler, MinMaxScaler
import xgboost as xgb


Scaler = StandardScaler()
PCAselector = PCA(0.99, svd_solver='full', random_state=1)
FilterSelector = SelectKBest(f_classif)

# clf = SVR('rbf')
clf = xgb.XGBRegressor(random_state=42) #objective="reg:linear",

PL = Pipeline(steps=[
        ('Scaler', Scaler),
        # ('PCAselector', PCAselector),
        # ('FilterSelector', FilterSelector),
        ('clf', clf)
        ])

param_grid={
    'clf__alpha': np.linspace(0,1,3),
    'clf__learning_rate': np.linspace(0,1,3),
    # 'clf__max_depth': np.linspace(0,1,10),
    # 'clf__subsample': 1,
    'clf__colsample_bytree': np.linspace(0,1,3),
    # 'clf__n_estimators': np.linspace(0,1,10)
    } #np.arange(0.1,1.1,0.1) 'kernel': ('linear', 'rbf')
est = GridSearchCV(
    PL,
    param_grid=param_grid,
    cv=TimeSeriesSplit(),
    n_jobs=1,
    scoring='neg_root_mean_squared_error')

est.fit(x_train,y_train) #,**{'ExtraTrees__sample_weight': weights}
final_model = est.best_estimator_
plt.plot(est.cv_results_['mean_test_score'])

#%%
y_pred = final_model.predict(x_train)
mean_squared_error(y_train,y_pred)
plt.close()
plt.subplot(121)
plt.plot(y_pred,'b',label='y_pred')
plt.plot(y_train,'r',label='y_true')
plt.legend()
y_pred = final_model.predict(x_test)
mean_squared_error(y_test,y_pred)
plt.subplot(122)
plt.plot(y_pred,'b',label='y_pred')
plt.plot(y_test,'r',label='y_true')
plt.legend()
# for i in range(len(y_pred)):
#     if y_pred[i]>=0.8:
#         print('sell',i)
#     if y_pred[i]<=-0.8:
#         print('buy',i)




#%%
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.utils import get_custom_objects

# source_model = VGG16() #include_top=False,input_tensor=Input(shape=(100, 100, 3))
source_model = VGG16(weights="imagenet")
# source_model.summary()

model = Sequential()
model.add(Input(shape=(50,50,3)))
for layer in source_model.layers[:-3]: # go through until last layer
    model.add(layer)

model.add(Dense(1024, activation='relu'))
model.add(Dense(3, activation='softmax'))
# model.summary()

for layer in model.layers[:-3]:
   layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam())

# best_model_path = os.path.join(OUTPUT_PATH, 'best_model_keras')
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
#                    patience=50, min_delta=0.0001)
# rlp = ReduceLROnPlateau()
# mcp = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=0,
#                       save_best_only=True, save_weights_only=False, mode='min', period=1)


history = model.fit(x_train, y_train, epochs=100, verbose=1,
                    batch_size=100, shuffle=True,
                    validation_data=(x_cv, y_cv))#,
                    # callbacks=[es, mcp, rlp])#,

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Metrics')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper right')

#%%
koers_test = df['close'][round(df.shape[0]*0.7)+window:].to_numpy()
plt.figure(figsize=(10,10))
plt.plot(koers_test,'b-')


idx_buy = np.where(np.argmax(y_test,1)==1)[0]
idx_sell = np.where(np.argmax(y_test,1)==0)[0]
plt.plot(idx_buy,koers_test[idx_buy],'r*')
plt.plot(idx_sell,koers_test[idx_sell],'g*')