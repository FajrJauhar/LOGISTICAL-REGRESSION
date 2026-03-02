import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("Phishing_Legitimate_full.csv")
#print(df.head())
#print(df.tail())

#Features
X =  df.drop(columns=['CLASS_LABEL','id'])
#Target
y = df['CLASS_LABEL']

#TRAINING
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled.mean(),X_test_scaled.std())

model = LogisticRegression(class_weight= 'balanced',
                           random_state= 42,
                           max_iter=1000)
model.fit(X_train_scaled,y_train)