import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import tree
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import datetime as dt



train = pd.read_csv("/home/machine/Desktop/testtest.csv")
# train=pd.read_csv("/home/machine/Desktop/test_X2.csv")
test = pd.read_csv("/home/machine/Desktop/test_22.csv")
output_y = pd.read_csv("/home/machine/Desktop/test_23.csv")
# train = train_data.append(test_data)
print(test.columns)

numerical_cols = ['age', 'antiguedad', 'renta']

feature_cols = ['ind_actividad_cliente',
                "ind_empleado", "pais_residencia" ,"sexo" , "ind_nuevo",
                 "nomprov", "segmento", 'indrel', 'tiprel_1mes', 'indresi', 'indext',
               'conyuemp', 'indfall', 'canal_entrada']


target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
               'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1',
               'ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
               'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
               'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']



#
# print(train.shape)
# print()

# train = train[np.isfinite(train['renta'])]
#
# print(train.shape)
train = train.dropna(subset=['antiguedad'])
train=train[train['antiguedad']!='-999999']
train=train[train['antiguedad']!='     NA']
# print(train['antiguedad'])
# print(train['antiguedad'].unique())
train=train.dropna(subset = ['ind_empleado'])
train=train.dropna(subset=['canal_entrada'])
train=train.dropna(subset=['indfall'])
train=train.dropna(subset=['tipodom'])
train=train.dropna(subset=['nomprov'])
train['renta'] = train['renta'].fillna(train.groupby(['nomprov'])['renta'].transform('mean'))
train['conyuemp'].fillna(0, inplace=True)
print(test['antiguedad'].unique())
test = test.dropna(subset=['antiguedad'])
test=test.dropna(subset = ['ind_empleado'])
test=test.dropna(subset=['canal_entrada'])
test=test.dropna(subset=['indfall'])
test=test.dropna(subset=['tipodom'])
test=test.dropna(subset=['nomprov'])
test['renta'] = test['renta'].fillna(test.groupby(['nomprov'])['renta'].transform('mean'))
test['conyuemp'].fillna(0, inplace=True)





for ind, col in enumerate(feature_cols):
    #print(train[col].dtype)
    if train[col].dtype == "object":
        le = LabelEncoder()
        le.fit(list(train[col].values) + list(test[col].values))
        temp_train_X_c = le.transform(list(train[col].values)).reshape(-1,1)
        temp_test_X_c = le.transform(list(test[col].values)).reshape(-1, 1)

    else:
        temp_train_X_c = np.array(train[col]).reshape(-1,1)
        temp_test_X_c = np.array(test[col]).reshape(-1, 1)


    ##################### fill NAN into most frequent value########################
    imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

    imr = imr.fit(temp_train_X_c)
    imr2 = imr.fit(temp_test_X_c)

    data = imr.transform(temp_train_X_c)
    data_test=imr2.transform(temp_test_X_c)

    if ind == 0:
        train_X_C = data.copy()
        test_X_C = data_test.copy()

    else:
        train_X_C = np.hstack([train_X_C,data])
        test_X_C = np.hstack([test_X_C,data_test])

#print(np.isnan(train_X_C[:,:]).sum())

#################################### Categorical data done ############################################################




train['fecha_dato_dt']=pd.to_datetime(train['fecha_dato'])
train['fecha_alta_dt']=pd.to_datetime(train['fecha_alta'])
train['time_difference']=(train['fecha_dato_dt']-train['fecha_alta_dt']).dt.days

test['fecha_dato_dt']=pd.to_datetime(test['fecha_dato'])
test['fecha_alta_dt']=pd.to_datetime(test['fecha_alta'])
test['time_difference']=(test['fecha_dato_dt']-test['fecha_alta_dt']).dt.days
numerical_cols.append('time_difference')

############################### new feature created : Current date - Open date #########################



for ind, col in  enumerate(numerical_cols):

        #train[col].fillna(-1,inplace=True)
        if train[col].dtype == "object":
            temp_train_X_n = pd.to_numeric(train[col], 'coerce').astype('float64').reshape(-1, 1)
            temp_test_X_n = pd.to_numeric(test[col], 'coerce').astype('float64').reshape(-1, 1)

        else:
            temp_train_X_n= np.array(pd.to_numeric(train[col], 'coerce').astype('float64')).reshape(-1, 1)
            temp_test_X_n = np.array(pd.to_numeric(test[col], 'coerce').astype('float64')).reshape(-1, 1)

        imr = Imputer(missing_values='NaN', strategy='mean', axis=0)

        imr = imr.fit(temp_train_X_n)
        imr2 =imr.fit(temp_test_X_n)

        data2 = imr.transform(temp_train_X_n)
        data2_test = imr2.transform(temp_test_X_n)

        if ind == 0:
            train_X_N = data2.copy()
            test_X_N = temp_test_X_n.copy()
        else:
            train_X_N = np.hstack([train_X_N, data2])
            test_X_N = np.hstack([test_X_N,data2_test])

#########################################################Numerical category done######################################################

full_train = np.hstack((train_X_C,train_X_N))
# print(train_X_C.shape ,train_X_N.shape , test_X_C.shape, temp_test_X_n.shape)
#
# print(test_X_C.shape)
# print('full_train',full_train.shape)
full_test = np.hstack((test_X_C,test_X_N))
print(full_test.shape)

result={}
for i in range(1000):
    result[i] = {}
import copy
for ind,col in enumerate(target_cols):

    temp_target_cols = copy.deepcopy(target_cols)
    del(temp_target_cols[ind])
    temp_train_X_T = train[temp_target_cols]
    temp_test_X_T = test[temp_target_cols]
    train_X_T = np.array(temp_train_X_T.fillna(0).astype('int'))
    test_X_T =np.array(temp_test_X_T.fillna(0).astype('int'))
    # print(train_X_T)
    full_train2=np.hstack((full_train,train_X_T))
    print(full_test.shape)
    print(test_X_T.shape)
    full_test2=np.hstack((full_test,test_X_T))

    #
    # print('full_train2',full_train2.shape)
    # print('full_test',full_test.shape)

    # print(col)
    train_y = train[col]
    train_y = np.array(train_y.fillna(0)).astype('int')
    train_X = full_train2
    test_X = full_test2
    # # print(full_train2.shape)
    # print(train_X.shape)
    # print(test_X.shape)
    # # test_y = pd.read_csv("/home/machine/Desktop/output_y.csv")
    # # test_y = np.array(test_y.astype('int'))
    #
    # # train_X,test_X,train_y,test_y = train_test_split(full_train2,train_y,test_size=0.3,random_state=0)
    sc=StandardScaler()
    sc2=StandardScaler()
    sc.fit(train_X)
    sc2.fit(test_X)
    X_train_std = sc.transform(train_X)
    X_test_std = sc2.transform(test_X)

    #
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    param_grid = {'max_depth': np.arange(3, 10)}
    bdt = GridSearchCV(DecisionTreeClassifier(), param_grid)
    bdt.fit(X_train_std, train_y)
    y_pred = bdt.predict(X_test_std)
    # bdt= AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
    #     n_estimators=1,
    #     learning_rate=0.1)
    # bdt.fit(X_train_std,train_y)
    # y_pred=bdt.predict(X_test_std)
    # print(y_pred)
    predict=bdt.predict_proba(X_test_std)
    print(bdt.predict_proba(X_test_std))
    print("ind!!!",ind)

    for i in range(1000):
        result[i][col]=1-predict[i][0]

    print(bdt.predict_proba(X_test_std))
    #
    # print('classified smaple : %d'%(test_y==y_pred).sum())
    # # # print('Miscalssified samples : %d' %(test_y!=y_pred).sum())
    # # # print('Accuracy : %.4f' %accuracy_score(test_y,y_pred))
    # # # print('-----------------------------------------------------------------------------')
import operator
for i in range(1000):
    result[i]=sorted(result[i].items(), key=operator.itemgetter(1), reverse=True)
f = open("/home/machine/Desktop/output1.txt",'w')

for i in range(1000):
    for j in range(24):
        line=str(result[i][j][0]) +' '
        f.write(line)
    f.write("\n")