# -*- coding: utf-8 -*-
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
from IPython.display import clear_output

def dataXY(df):
    def create_labels(series, window, pred_future):
#        prominence=0
#        fpmax = find_peaks(series,prominence=prominence)[0]
#        fpmin = find_peaks(-series,prominence=prominence)[0]
#        y = np.array(series.shape[0]*[2])
#        y = np.expand_dims(y,-1)
#        y[fpmax,:] = 0
#        y[fpmin,:] = 1
#        y = y[window:,:]
        
        y = ((series.diff(periods=pred_future)>0)*1).to_numpy()
        y = y[window+pred_future:]
#        y = np.squeeze(y)
#        y = np.expand_dims(y[window+pred_future:],-1)
        
#        y = OneHotEncoder(sparse=False, categories='auto').fit_transform(y)
        return y

    def GAF(series,window,image_size,pred_future):
        gasf = GramianAngularField(image_size=image_size, method='difference')
        
        X_gasf = np.zeros((series.shape[1]-window-pred_future,image_size,image_size,3))
        
        for i in range(series.shape[1]-window-pred_future):
            step = 1
            series_temp = series[:,np.linspace(i,(window-1)*step+i,window,dtype=np.int)]
            gasf_g = np.moveaxis(gasf.transform(series_temp), (0), (-1))
            gasf_g = np.squeeze(gasf_g)
#            cmap = plt.get_cmap('jet')
#            rgba_img = cmap(gasf_g)
#            rgb_img = np.delete(rgba_img, 3, 2)
#            plt.imshow(rgb_img)
            X_gasf[i,:,:,:] = gasf_g
            
            
        
#        X_gasf = np.zeros((series.shape[1]-((window-1)*10),image_size*2,image_size*2,series.shape[0]))
#        
#        for i in range(0,series.shape[1]-((window-1)*10)):
#            step = 1
#            series_temp = series[:,np.linspace(i,(window-1)*step+i,window,dtype=np.int)]
#            X_gasf[i,0:image_size,0:image_size,:] = np.moveaxis(gasf.transform(series_temp), (0), (-1))
#            step = 3
#            series_temp = series[:,np.linspace(i,(window-1)*step+i,window,dtype=np.int)]
#            X_gasf[i,image_size:,0:image_size,:] = np.moveaxis(gasf.transform(series_temp), (0), (-1))
#            step = 5
#            series_temp = series[:,np.linspace(i,(window-1)*step+i,window,dtype=np.int)]
#            X_gasf[i,0:image_size,image_size:,:] = np.moveaxis(gasf.transform(series_temp), (0), (-1))
#            step = 10
#            series_temp = series[:,np.linspace(i,(window-1)*step+i,window,dtype=np.int)]
#            X_gasf[i,image_size:,image_size:,:] = np.moveaxis(gasf.transform(series_temp), (0), (-1))
            
        return X_gasf

    train_size = 0.5
    cv_size = 0.3
    window = 50
    image_size = 40
    pred_future = 1
    
    
    b = abs(df.iloc[:,[2,3]].diff(axis=1)).iloc[:,1:2].to_numpy()
    c = df.iloc[:,[4,6]].to_numpy()
    d = np.concatenate((b,c),1)
#    d = df.iloc[:,4:5].to_numpy()

    e = d[0:round(df.shape[0]*train_size),:].T
    x_train = GAF(e,window,image_size,pred_future)
    y_train = create_labels(df['high'][0:round(df.shape[0]*train_size)], window, pred_future)

    e = d[round(df.shape[0]*train_size):round(df.shape[0]*(train_size+cv_size)),:].T
    x_cv = GAF(e,window,image_size,pred_future)
    y_cv = create_labels(df['high'][round(df.shape[0]*train_size):round(df.shape[0]*(train_size+cv_size))], window, pred_future)

    e = d[round(df.shape[0]*(train_size+cv_size)):,:].T
    x_test = GAF(e,window,image_size,pred_future)
    y_test = create_labels(df['high'][round(df.shape[0]*(train_size+cv_size)):], window, pred_future)

    return x_train, y_train, x_cv, y_cv, x_test, y_test


df = pd.read_csv("C:/Users/beukingarj/Downloads/stock_cnn_blog_pub-master/stock_cnn_blog_pub-master/stock_history/WMT/WMT.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)
df.reset_index(inplace=True,drop=True)
x_train, y_train, x_cv, y_cv, x_test, y_test = dataXY(df)

window=50
koers_test = df['high'][round(df.shape[0]*0.8)+window:].to_numpy()
plt.figure(figsize=(10,10))
plt.plot(koers_test[0:100],'b-')

idx_higher = np.where(y_test[0:100])[0]
idx_lower = np.where(y_test[0:100]==0)[0]
plt.plot(idx_higher,koers_test[idx_higher],'r*',label='higher')
plt.plot(idx_lower,koers_test[idx_lower],'g*',label='lower')
plt.title("y_test")
plt.legend()


#%%
source_model = VGG16(include_top=False,input_tensor=Input(shape=x_train.shape[1:]))
#weights="imagenet",
last = source_model.output
x = Flatten()(last)
x = Dense(1024, activation='relu')(x)
preds = Dense(2, activation='sigmoid')(x)
model = Model(source_model.input, preds)

#for layer in model.layers[:-3]:
#   layer.trainable = False

model.compile(loss='binary_crossentropy', optimizer='adam')

best_model_path = "C:/Users/beukingarj/Downloads/stock_cnn_blog_pub-master/stock_cnn_blog_pub-master/outputs/cnn_jorn/weights.hdf5"
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=0, patience=10)
rlp = ReduceLROnPlateau()
mcp = ModelCheckpoint(best_model_path, save_best_only=True)

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_acc'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="accuracy")
        plt.plot(self.x, self.val_losses, label="val_accuracy")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show();

plot_losses = PlotLosses()

_,class_count = np.unique(np.argmax(y_train,1),return_counts=True)
history = model.fit(x_train, y_train, epochs=50, verbose=0,
                    batch_size=30, shuffle=True,
                    validation_data=(x_cv, y_cv),
                    callbacks=[plot_losses], #, es, mcp, rlp
                    class_weight={0: class_count[1]/sum(class_count), 1: class_count[0]/sum(class_count)})

model.load_weights(best_model_path)
predicted = model.predict(x_test)



#%%

window=50
koers_test = df['close'][round(df.shape[0]*0.8)+window:].to_numpy()
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.plot(koers_test[0:100],'b-')

idx_higher = np.where(y_test[0:100])[0]
idx_lower = np.where(y_test[0:100]==0)[0]
plt.plot(idx_higher,koers_test[idx_higher],'r*',label='higher')
plt.plot(idx_lower,koers_test[idx_lower],'g*',label='lower')
plt.title("y_test")
plt.legend()

plt.subplot(122)
plt.plot(koers_test[0:100],'b-')

y_pred = model.predict(x_test[0:100])
idx_buy = np.where(np.argmax(y_pred,1)==1)[0]
idx_sell = np.where(np.argmax(y_pred,1)==0)[0]
plt.plot(idx_buy,koers_test[idx_buy],'r*')
plt.plot(idx_sell,koers_test[idx_sell],'g*')
plt.title("y_pred")


#%%

from ta import add_all_ta_features

def dataXY(df,train_size=0.7,cv_size=0,window=50,pred_future=30):
    def create_labels(y_cohort, window, pred_future):
        y = (y_cohort.diff(periods=pred_future).shift(-pred_future-window).dropna()>=0).astype(int)
        return y

    def create_ta(x_cohort,window,pred_future):
        x = add_all_ta_features(x_cohort,'open','high','low','close','volume',fillna=True).shift(-pred_future).dropna()
        return x 
    
    train_cohort = df[0:round(df.shape[0]*train_size)]
    x_train_cohort = train_cohort.iloc[:,1:7]
    x_train = create_ta(x_train_cohort, window, pred_future)
    y_train_cohort = train_cohort['close']
    y_train = create_labels(y_train_cohort, window, pred_future)
    
#    cv_cohort = df[round(df.shape[0]*train_size):round(df.shape[0]*(train_size+cv_size))]
#    x_cv_cohort = cv_cohort.iloc[:,1:7]
#    x_cv = create_ta(x_cv_cohort, window, pred_future)
#    y_cv_cohort = cv_cohort['close']
#    y_cv = create_labels(y_cv_cohort, window, pred_future)
    
    test_cohort = df[round(df.shape[0]*(train_size+cv_size)):]
    x_test_cohort = test_cohort.iloc[:,1:7]
    x_test = create_ta(x_test_cohort, window, pred_future)
    y_test_cohort = test_cohort['close']
    y_test = create_labels(y_test_cohort, window, pred_future)
    y_test_cohort = y_test_cohort.shift(-pred_future).dropna()
    
    return x_train, y_train, x_cv, y_cv, x_test, y_test, y_test_cohort

df = pd.read_csv("C:/Users/beukingarj/Downloads/stock_cnn_blog_pub-master/stock_cnn_blog_pub-master/stock_history/WMT/WMT.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)
df.reset_index(inplace=True,drop=True)
x_train, y_train, x_cv, y_cv, x_test, y_test, y_test_cohort = dataXY(df,window=0,pred_future=10)


#%%
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoLarsIC
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, classification_report, confusion_matrix, precision_score
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


plt.close()

parameters = {
#    'clf__base_estimator__n_estimators': np.round(np.linspace(100,400,10)).astype('int'),
#    'clf__base_estimator__max_depth': [10,11,12],
#    'clf__base_estimator__min_child_weight': [1],
#    'clf__base_estimator__gamma': np.linspace(0,0.5,5),
#    'clf__base_estimator__subsample': np.linspace(0.2,0.4,3),
#    'clf__base_estimator__colsample_bytree': np.linspace(0.2,0.4,3),    
#    'clf__base_estimator__reg_alpha': np.linspace(0.01,0.03,10)
#    'clf__method': ['isotonic','sigmoid'],
}

class LASSOJorn( BaseEstimator, TransformerMixin ):
    def __init__( self ):
        None
    
    def fit( self, X, y ):
        self.model = LassoLarsIC(criterion='aic').fit(X, y)
        return self
    
    def transform( self, X ):
        return np.asarray(X)[:,abs(self.model.coef_)>0]

eval_set = [(x_train, y_train), (x_cv, y_cv)]
scale_pos_weight=Counter(y_train)[0]/Counter(y_train)[1]
clf = xgb.XGBRFClassifier(objective='binary:logistic', 
                          scale_pos_weight=scale_pos_weight,
                          learning_rate=0.01,
                          n_estimators=5000,
                          max_depth=10,
                          min_child_weight=1,
                          gamma=0,
                          subsample=0.3,
                          colsample_bytree=0.3,
                          reg_alpha=0.014,
                          nthread=4,
                          seed=27
                          )

PL = Pipeline(steps=[('PreProcessor', StandardScaler()),
                     ('PCA', PCA()),
                     ('EmbeddedSelector', LASSOJorn()),
                     ('clf', CalibratedClassifierCV(base_estimator=clf,method='sigmoid'))])

#tss = TimeSeriesSplit(n_splits=3)
#optimizer = GridSearchCV(PL, parameters, cv=tss, n_jobs=-1, verbose=10, scoring='roc_auc')
#optimizer.fit(x_train, y_train)
#print(optimizer.best_params_)
#final_model = optimizer.best_estimator_
final_model = PL.fit(x_train, y_train)

#plt.plot(optimizer.cv_results_['mean_test_score'])
#xgb.plot_importance(final_model.named_steps['clf'])

y_pred_proba = final_model.predict_proba(x_test)[:,1]
y_pred = final_model.predict(x_test)

fraction_of_positives, mean_predicted_value = calibration_curve(np.array(y_test), y_pred_proba, strategy='uniform',n_bins=20)
plt.figure()
plt.plot(mean_predicted_value, fraction_of_positives, "sr-")
plt.title("Calibration")
plt.xlabel("mean_predicted_value")
plt.ylabel("fraction_of_positives")

fpr, tpr, _ = roc_curve(y_test,y_pred_proba)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
plt.title("ROC")

range_class = np.arange(0.1,0.9,0.01)
PPV = np.zeros(len(range_class))
NPV = np.zeros(len(range_class))
j=0
for i in range_class:
    PPV[j] = precision_score(y_test, y_pred_proba>i,pos_label=1)
    NPV[j] = precision_score(y_test, y_pred_proba>i,pos_label=0)
    j+=1
plt.figure()    
plt.plot(range_class,PPV,label='PPV')
plt.plot(range_class,NPV,label='NPV')
plt.legend()
threshold = 0.98
threshold_high = range_class[np.where(PPV>threshold)[0][0]]
threshold_low = range_class[np.where(NPV<threshold)[0][0]]
plt.plot(threshold_high,PPV[np.where(np.isin(range_class,threshold_high))[0][0]],'r*')
plt.plot(threshold_low,NPV[np.where(np.isin(range_class,threshold_low))[0][0]],'r*')

plt.figure(figsize=(10,10))
idx = np.linspace(0,100,101).astype('int')
plt.plot(range(len(y_test_cohort.iloc[idx])),y_test_cohort.iloc[idx],'b')
idx_high = np.where(y_pred_proba[idx]>threshold_high)[0]
plt.plot(idx_high,np.asarray(y_test_cohort)[idx_high],'g.')
idx_low = np.where(y_pred_proba[idx]<threshold_low)[0]
plt.plot(idx_low,np.asarray(y_test_cohort)[idx_low],'r.')

idx_sure = np.sort(np.concatenate((idx_high,idx_low)))
print(classification_report(y_test.iloc[idx_sure],y_pred[idx_sure]))
print(confusion_matrix(y_test.iloc[idx_sure],y_pred[idx_sure]))


#%%

def bot(threshold_high,threshold_low):
    koers=df.iloc[round(df.shape[0]*(0.7)):,4]
    
    start=np.zeros(len(x_test)+1)
    start[0]=20000
    bought=0
    sellat=0
    buyat=0
    for i in range(len(x_test)):
        if y_pred_proba[i]>threshold_high and bought==0:
#            print("Buy at i=",i)
            buyat = koers.iloc[i]
            
            if sellat!=0:
                interest = -(buyat-sellat)/sellat
                start[i+1] = start[i]*(1+interest)
            else:
                start[i+1]=start[i]
            
            bought=1
        elif y_pred_proba[i]<threshold_low and bought==1:
#            print("Sell at i=",i)
            sellat = koers.iloc[i]
            
            if buyat!=0:
                interest = (sellat-buyat)/buyat
                start[i+1] = start[i]*(1+interest)
            else:
                start[i+1]=start[i]
            bought=0
            
            
        else:
            start[i+1]=start[i]
    
    return start


#range_class = np.arange(0.1,1,0.01)
#interest = np.zeros((range_class.shape[0],range_class.shape[0]))
#ii=0
#for i in range_class:
#    jj=0
#    for j in range_class:
#        start = bot(i,j)
#        interest[ii,jj] = start[-1]/start[0]*100/len(start)
#        jj+=1
#    ii+=1
#ind = np.unravel_index(np.argmax(interest), interest.shape)

start = bot(0.38,0.13)
interest = start[-1]/start[0]*100/len(start)
print("interest: ",interest)
plt.plot(start[0:])



#%%

a=np.zeros(len(parameters.keys()))
j=0
for key in parameters.values():
    a[j] = np.round(len(key))
    j+=1
a = a.astype('int')
reshaped_scores = np.reshape(optimizer.cv_results_['mean_test_score'],a)
reshaped_scores = np.squeeze(reshaped_scores)

plt.subplot(221)
plt.plot(parameters['clf__n_estimators'],reshaped_scores[:,0,0,0])
plt.subplot(222)
plt.plot(parameters['clf__learning_rate'],reshaped_scores[0,:,0,0])
plt.subplot(223)
plt.plot(parameters['clf__max_depth'],reshaped_scores[0,0,:,0])
plt.subplot(224)
plt.plot(parameters['clf__gamma'],reshaped_scores[0,0,0,:])



#%%

# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227}




