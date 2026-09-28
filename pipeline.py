from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
import joblib

df = pd.read_csv('./data/insurance.csv')
features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

X = df[features]
Y = df['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
categorical_features = ['sex', 'smoker', 'region']

categorical_transformer = OneHotEncoder(handle_unknown="ignore")
preprocessor = ColumnTransformer([("categorical", categorical_transformer, categorical_features)], remainder='passthrough')

pipeline = Pipeline([('Preprocessing', preprocessor), ('Polynomial', PolynomialFeatures(2)), ('Model', LinearRegression())])
pipeline.fit(X_train, Y_train)
y_hat = pipeline.predict(X_test)

joblib.dump(pipeline, 'model.joblib')