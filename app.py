# Importing Necessary Libraries
from posixpath import split
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from flask import Flask, render_template, request

webapp=Flask(__name__)


@webapp.route('/')
def index():
    return render_template('index.html')

@webapp.route('/about')
def about():
    return render_template('about.html')

@webapp.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@webapp.route('/view')
def view():
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())






@webapp.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  countvectorizer
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100

        print('#########################################')

        df = pd.read_csv("spam (1).csv", encoding='latin-1')

        stemming = PorterStemmer()
        corpus = []
        for i in range (0,len(df)):
            s1 = re.sub('[^a-zA-Z]',repl = ' ',string = df['Message'][i])
            s1.lower()
            s1 = s1.split()
            s1 = [stemming.stem(word) for word in s1 if word not in set(stopwords.words('english'))]
            s1 = ' '.join(s1)
            corpus.append(s1)
        print('###########################################')
        print(corpus[50]) 
        print('###########################################')
        df["Category"] = np.where(df["Category"] == "ham", 0, 1) 
        print(df)
        print('#########################################')
        # countvectorizer =CountVectorizer()
        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False)
        x = hvectorizer.fit_transform(corpus).toarray()
        print(x)

        y = df['Category'].values
        print(y)
        

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=size, random_state=42)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')


@webapp.route('/model',methods=['POST','GET'])
def model():

    if request.method=="POST":
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg='Please Choose an Algorithm to Train')
        elif s==1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            multinomialnb = MultinomialNB()
            multinomialnb.fit(x_train,y_train)
            # Predicting the Test set results
            y_pred = multinomialnb.predict(x_test)
            acc_rf = multinomialnb.score(x_test, y_test)*100
            from sklearn.metrics import confusion_matrix
            print(confusion_matrix(y_test, y_pred))
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Naive Bayes Classifier is ' + str(acc_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s==2:
            linearsvc = LinearSVC()
            linearsvc.fit(x_train,y_train)
            y_pred = linearsvc.predict(x_test)
            acc_dt = linearsvc.score(x_test, y_test)*100
            from sklearn.metrics import confusion_matrix
            print(confusion_matrix(y_test, y_pred))
            msg = 'The accuracy obtained by Support Vector Classifier is ' + str(acc_dt) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')

@webapp.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)
        
        # countvectorizer =CountVectorizer()
        multinomialnb = MultinomialNB()
        multinomialnb.fit(x_train,y_train)
        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False)
        # from sklearn.feature_extraction.text import CountVectorizer
        # countvectorizer =CountVectorizer()
        result =multinomialnb.predict(hvectorizer.transform([f1]))
        if result==0:
            msg = 'This is a Ham Message'
        else:
            msg= 'This is a Spam Message'
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')

@webapp.route('/news')
def news():
    return render_template('news.html')



if __name__=='__main__':
    webapp.run(debug=True)