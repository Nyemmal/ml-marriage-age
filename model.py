import joblib
import pandas as pd
data = pd.read_csv('age_of_marriage_data.csv')
print(data.shape)

data.head()
(data.shape[0] - data.dropna().shape[0])/data.shape[0]
data.dropna(inplace=True)
data.shape
data.head(2)
data.profession.unique()

X = data.loc[:,['gender','height','religion','caste','mother_tongue','country']]
y = data.age_of_marriage

X.head()
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
X.loc[:,['gender','religion','caste','mother_tongue','country']]= \
X.loc[:,['gender','religion','caste','mother_tongue','country']].apply(enc.fit_transform)

X.head()
int(X.loc[1,'height'].split('\'')[0])*30.48
int(X.loc[1,'height'].split('\'')[1].replace('"',''))*2.54
def h_cms(h):
    return int(h.split('\'')[0])*30.48+\
    int(h.split('\'')[1].replace('"',''))*2.54
X['height_cms'] = X.height.apply(h_cms)

X.head()
X.drop('height',inplace=True,axis=1)
X.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=80,max_depth=11)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score
print("MAE : ", mean_absolute_error(y_test,y_predict))
r2_score(y_test,y_predict)

import pickle
pickle.dump(model, open('model.pkl','wb'))

#Prediction
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0,5,4,4,6,175]]))