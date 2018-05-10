import pandas as pd
from sklearn.naive_bayes import MultinomialNB
feature_data = pd.read_csv("feature_data.csv")

features = pd.read_csv("features.csv")
trial = pd.concat([feature_data,features],axis=1)
trial = trial.dropna(axis=0,how="any")



# print feature_data

# print feature_data["keyword_or_not"]

# print features.as_matrix()

clf = MultinomialNB(alpha=0.8)
a = clf.fit(trial.drop(["word","keyword_or_not"],axis=1),trial["keyword_or_not"])
print a.predict(features[77:87])