from sklearn.linear_model import LogisticRegression

model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Logistic Accuracy:", accuracy_score(y_test, pred2))
