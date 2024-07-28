import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv('data/intent_text.csv')
    df_response = pd.read_csv('data/intent_response.csv')
    return df, df_response

def preprocess_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['intent'], test_size=0.2, random_state=42)
    
    intent_to_index = {intent: index for index, intent in enumerate(df['intent'].unique())}
    y_train_encoded = y_train.map(intent_to_index)
    y_test_encoded = y_test.map(intent_to_index)
    
    return X_train, X_test, y_train_encoded, y_test_encoded, intent_to_index