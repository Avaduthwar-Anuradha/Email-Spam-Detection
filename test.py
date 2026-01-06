import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens
'''
dataset = pd.read_csv("Dataset/spam_ham_dataset.csv",encoding='iso-8859-1')
print(dataset.shape)
for i in range(len(dataset)):
    msg = dataset.get_value(i, 'text')
    label = int(dataset.get_value(i, 'label_num'))
    msg = str(msg)
    msg = msg.strip().lower()
    labels.append(label)
    clean = cleanPost(msg)
    textdata.append(clean)
    print(i)

labels = np.asarray(labels)
textdata = np.asarray(textdata)
np.save("labels.txt",labels)
np.save("email.txt",textdata)

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=500)
tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
print(str(df))
print(df.shape)
df = df.values
X = df[:, 0:500]
Y = np.asarray(labels)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

np.save("Y.txt",Y)
np.save("X.txt",X)
with open('model/tfidf.txt', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)
file.close()
'''
Y = np.load("model/Y.txt.npy")
X = np.load("model/X.txt.npy")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

with open('model/tfidf.txt', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
file.close()


if os.path.exists('model/rf.txt'):
    with open('model/rf.txt', 'rb') as file:
        rf = pickle.load(file)
    file.close()
else:
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    with open('model/rf.txt', 'wb') as file:
        pickle.dump(rf, file)
    file.close()
        

predict = rf.predict(X_test)
acc = accuracy_score(y_test,predict)*100
p = precision_score(y_test,predict,average='macro') * 100
r = recall_score(y_test,predict,average='macro') * 100
f = f1_score(y_test,predict,average='macro') * 100
print(str(acc)+" "+str(p)+" "+str(r)+" "+str(f))


dataset = pd.read_csv("Dataset/testMessages.txt",encoding='iso-8859-1',nrows=10)
for i in range(len(dataset)):
    msg = dataset.get_value(i, 'email')
    msg = str(msg)
    msg1 = msg.strip().lower()
    clean = cleanPost(msg1)
    tfidf = tfidf_vectorizer.transform([clean]).toarray()
    predict = rf.predict(tfidf)
    print(msg1+"======"+str(predict))



