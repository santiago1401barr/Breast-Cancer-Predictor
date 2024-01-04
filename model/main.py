import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle

def create_model(data): 
  X = data.drop("diagnosis", axis = 1)
  y = data["diagnosis"]
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
  X_train = scale(X_train)
  X_test = scale(X_test)
  scaler, model  = make_pipeline(StandardScaler(), LogisticRegression())
  model .fit(X_train, y_train)

  return model, scaler

def get_clean_data():
  data = pd.read_csv("data/data.csv")
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  data["diagnosis"] = data["diagnosis"].map({ "M" : 1, "B": 0 })

  return data

def main():
  data = get_clean_data()
  model, scaler = create_model(data)
  
  with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
  with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
if __name__ == '__main__':
  main()
