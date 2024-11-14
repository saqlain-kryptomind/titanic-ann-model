import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(file_path):
    # Load Titanic dataset
    df = pd.read_csv(file_path)

    # Handle missing values: Fill missing 'Age' with mean, drop rows with missing 'Embarked' and 'Survived'
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df.dropna(subset=['Embarked', 'Survived'], inplace=True)

    # Encode categorical features (Sex, Embarked)
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

    # Feature selection: We'll use 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
    y = df['Survived'].values.reshape(-1, 1)  # Ensure y is a 2D array

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
