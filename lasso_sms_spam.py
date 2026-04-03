import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message']).toarray()
y = df['label']

total_features = X.shape[1]
print("Total Features:", total_features)

def train(alpha):
    model = Lasso(alpha=alpha)
    model.fit(X, y)

    non_zero = np.sum(model.coef_ != 0)
    zero = np.sum(model.coef_ == 0)

    reduction = ((total_features - non_zero) / total_features) * 100

    print(f"\nAlpha: {alpha}")
    print("Selected Features:", non_zero)
    print("Removed Features:", zero)
    print("Reduction %:", round(reduction, 2))

train(0.01)
train(0.1)
train(1)
