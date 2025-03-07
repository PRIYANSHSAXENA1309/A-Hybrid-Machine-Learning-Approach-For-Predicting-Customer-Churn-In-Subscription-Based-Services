import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from scikeras.wrappers import KerasClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
import joblib

data = pd.read_csv("D:\Code\Subscription_Service_Churn_Dataset.csv")

data = data.drop(columns=['CustomerID'])

y = data['Churn']  # Assuming 'Churn' is the target column
X = data.drop(columns=['Churn'])

for col in X.select_dtypes(include=['float64', 'int64']).columns:
    X[col].fillna(X[col].median(), inplace=True)
for col in X.select_dtypes(include=['object']).columns:
    X[col].fillna(X[col].mode()[0], inplace=True)

label_encoder = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = label_encoder.fit_transform(X[col])

scaler = StandardScaler()
X[X.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(X.select_dtypes(include=['float64', 'int64']))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

def create_nn():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Defining models

random_forest = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

neural_network = KerasClassifier(build_fn=create_nn, epochs=20, batch_size=64, verbose=0)

logistic_regression = AdaBoostClassifier(estimator=LogisticRegression(), n_estimators=50, random_state=42)

naive_bayes = AdaBoostClassifier(estimator=GaussianNB(), n_estimators=50, random_state=42)

decision_tree_boosted = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

svc = BaggingClassifier(estimator=SVC(probability=True, kernel='rbf', C=1.0, gamma='scale'), n_estimators=50, random_state=42)


voting_clf = VotingClassifier(
    estimators=[
        ('rf', BaggingClassifier(estimator=random_forest, n_estimators=10, random_state=42)),
        ('knn', BaggingClassifier(estimator=knn, n_estimators=10, random_state=42)),
        ('nn', neural_network),
        ('log_reg', logistic_regression),
        ('nb', naive_bayes),
        ('dt_boost', decision_tree_boosted),
        ('svc', svc)
    ],
    voting='soft'
)


X_train = np.array(X_train)
y_train = np.array(y_train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in cv.split(X_train, y_train):
    X_cv_train, X_cv_val = X_train[train_index], X_train[val_index]
    y_cv_train, y_cv_val = y_train[train_index], y_train[val_index]
    
    voting_clf.fit(X_cv_train, y_cv_train)
    y_val_pred = voting_clf.predict(X_cv_val)
    print(f"AUC-ROC on fold: {roc_auc_score(y_cv_val, voting_clf.predict_proba(X_cv_val)[:, 1])}")


voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print("Final Test AUC-ROC:", roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1]))
print(classification_report(y_test, y_pred))

train_accuracy = voting_clf.score(X_train, y_train)
test_accuracy = voting_clf.score(X_test, y_test)

joblib.dump(voting_clf, "voting_classifier_model.pkl")
joblib.dump(scaler,"scaler.pkl")
