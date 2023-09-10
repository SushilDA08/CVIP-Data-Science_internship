#!/usr/bin/env python
# coding: utf-8

# # **<blockquote style="color:#0047AB; font-family: Arial, sans-serif;">DATA SCIENCE INTERNSHIP</blockquote>**
# 
# ## **<span style="color:#FF0000; font-family: Arial, sans-serif;">Task-4: EMAIL SPAM DETECTION</span>**
# 
# ### **Name:** SUSHIL PRASAD BOOPATHY M
# 
# ## **<span style="color:#00A300; font-family: Arial, sans-serif;">Problem Description</span>**
# 
# The Email Spam Identification Project seeks to address these challenges by developing a robust, accurate, and adaptable system that can efficiently differentiate between spam and legitimate emails while allowing users to customize their filtering preferences.
# 
# ## **<span style="color:#AA00FF; font-family: Arial, sans-serif;">Objective</span>**
# The "Spam Mail Detection" project seeks to deliver an intelligent, adaptable, and user-centric solution that effectively combats the persistent challenge of email spam, enhances email communication security, and improves the overall email experience for users across various platforms and languages.
# 
# 
# ![Spam_mail.jpg](attachment:Spam_mail.jpg)
# 

# ## <span style="color:#FF5733;">Importing the necessary libraries</span>

# In[251]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.colors as colors
import datetime as dt
from collections import Counter
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from tqdm.auto import tqdm
import time
from nltk.corpus import stopwords
from statistics import mode
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
get_ipython().system(' pip install --user xgboost')
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
get_ipython().system(' pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS

import warnings
warnings.filterwarnings("ignore")


# ## <span style="color:#FF5733;"> **Import & Reading Data File** </span>

# In[252]:


data = pd.read_csv('spam.csv',encoding = 'latin1')


# In[253]:


data.head(5)


# ## <span style="color:#FF5733;">Getting the dataset details</span>

# In[254]:


data.info()


# Dataset has 4 columns from these V1 and V2 are the necessary columns with non nul values
# other 2 unnamed columns were not necessary.

# ## <span style="color:#FF5733;">Dropping the unwanted columns </span>

# In[255]:


data.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace = True)


# In[256]:


data.sample(5)


# In[257]:


df = pd.DataFrame(data)
df


# ## <span style="color:#FF5733;">Renaming the columns  </span>

# In[258]:


df.rename(columns = {'v1':'Mail_type','v2':'Mails'},inplace = True)
df


# ## <span style="color:#FF5733;">checking the columns with null values  </span>

# In[259]:


df.isna().sum()


# ## <span style="color:#FF5733;"> Visualizing the mail_types by Bar and Pie chart </span>

# In[260]:


explode = (0.0, 0.2,) 

# Creating color parameters
colors = ( "cyan", "brown",)

Mail_type = ['Ham','Spam']
 
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "green" }
 
# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)
 
# Creating plot
fig, ax = plt.subplots(figsize =(6, 10))
wedges, texts, autotexts = ax.pie(df['Mail_type'].value_counts().sort_values(ascending=False),
                                  autopct = lambda pct: func(pct,df['Mail_type'].value_counts().sort_values(ascending=False)[:5]),
                                  explode = explode,
                                  labels = Mail_type,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="magenta"))
 
# Adding legend
ax.legend(wedges,Mail_type,
          title ="Recieved_Mail_types",
          loc ="center left",
          bbox_to_anchor =(1, 0, 0.5, 1))
 
plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("Types of mail recieved")
 
plt.show()

df['Mail_type'].value_counts().plot.bar(color = 'c')


# ## <span style="color:#FF5733;"> Retrieve the most repeated words in mails </span>

# In[261]:


Text = ' '.join(df['Mails'].values)
Text = re.sub(r'http\S+', '', Text)
Text = re.sub(r'@\S+', '', Text)
Text = re.sub(r'#\S+', '', Text)


# In[262]:


words = Text.split()


# In[263]:


stop_words = set(stopwords.words('english'))
words = [word for word in words if not word in stop_words]


# In[264]:


word_counts = Counter(words)
top_words = word_counts.most_common()
top_words


# In[329]:


top_words = word_counts.most_common(10)
x_values = [word[0] for word in top_words]
y_values = [word[1] for word in top_words]

fig = go.Figure(data=[go.Bar(x=x_values, y=y_values)])
fig.update_layout(title='Most repeated Words in Recieved mails', xaxis_title='Word', yaxis_title='Frequency')

fig.show()

comment_words = ''
stopwords = set(STOPWORDS)
 
# iterate through the data
for val in top_words:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "
    
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
# plot the WordCloud image                      
plt.figure(figsize = (10, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# ## <span style="color:#FF5733;"> Correcting the mails by word stemming,regular expression,stop words using NLP </span>

# In[266]:


def clean_text(text):
  text = re.sub('<.*?>', '', text)
  text = re.sub('[^a-zA-Z]', ' ', text).lower()
  words = nltk.word_tokenize(text)
  words = [w for w in words if w not in stopwords.words('english')]
  stemmer = PorterStemmer()
  words = [stemmer.stem(w) for w in words]
  text = ' '.join(words)
  return text


# In[267]:


get_ipython().run_cell_magic('time', '', "\ntqdm.pandas()\n\ndf['corrected_texts'] = df['Mails'].progress_apply(clean_text)\n")


# In[268]:


df['corrected_texts'][:10]


# In[269]:


df


# ## <span style="color:orange;">Applying Train-Test Split & Applying Regression Models, Evaluating Accuracy</span>

# ## <span style="font-family:candara">Executing a Train-Test Split,Applying <mark>Logistic Regression Classifier</mark>, and Evaluating Accuracy</span>

# <code style="background:yellow;color:black"> **Corected texts accuracy :** </code>

# In[270]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['corrected_texts']).toarray()

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[271]:


lr = LogisticRegression(random_state = 1)
model_lr = lr.fit(X_train,y_train)


# In[272]:


predict_lr = model_lr.predict(X_test)
print(f"Logistic Regression:" ,accuracy_score(y_test,predict_lr))


# In[273]:


y_pred_lr = lr.predict(X_test)

print(f" Logistic Regression Accuracy score:",accuracy_score(y_test, y_pred_lr))


# In[274]:


accuracy_lor = model_lr.score(X_test,y_test)
print(f"Logistic Regression Accuracy : {accuracy_lor}")


# <code style="background:black;color:cyan"> **Original mail texts accuracy :** </code>

# In[275]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['Mails'])

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[276]:


lr = LogisticRegression(random_state = 42)
model_lr = lr.fit(X_train,y_train)


# In[277]:


predict_lr = model_lr.predict(X_test)
print(f"Logistic Regression:" ,accuracy_score(y_test,predict_lr))


# In[278]:


accuracy_lor_org = model_lr.score(X_test,y_test)
print(f"Logistic Regression Accuracy : {accuracy_lor_org}")


# ## <span style="font-family:candara">Executing a Train-Test Split,Applying <mark>Decision Tree Classifier</mark>, and Evaluating Accuracy</span>

# <code style="background:yellow;color:black"> **Corected texts accuracy :** </code>

# In[279]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['corrected_texts']).toarray()

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[280]:


dt = DecisionTreeClassifier(random_state = 42)
model_dt = dt.fit(X_train,y_train)


# In[281]:


predict_dt = model_dt.predict(X_test)
print(f"Logistic Regression:" ,accuracy_score(y_test,predict_dt))


# In[282]:


accuracy_dtc = model_dt.score(X_test,y_test)
print(f"Random Forest Classifier Accuracy: {accuracy_dtc}")


# <code style="background:black;color:cyan"> **Original mail texts accuracy :** </code>

# In[283]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['Mails'])

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[284]:


dt = DecisionTreeClassifier(random_state = 42)
model_dt = dt.fit(X_train,y_train)


# In[285]:


predict_dt = model_dt.predict(X_test)
print(f"Random forest classifier:" ,accuracy_score(y_test,predict_dt))


# In[286]:


accuracy_dtc_org = model_dt.score(X_test,y_test)
print(f"Random Forest Classifier Accuracy: {accuracy_dtc_org}")


# ## <span style="font-family:candara">Executing a Train-Test Split,Applying <mark>Random Forest Classifier</mark>, and Evaluating Accuracy</span>

#  <code style="background:yellow;color:black"> **Corected texts accuracy :** </code>

# In[287]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['corrected_texts']).toarray()

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[288]:


rf= RandomForestClassifier(random_state=42)
model_rf = rf.fit(X_train, y_train)


# In[289]:


predict_rf = model_rf.predict(X_test)
print(f"Logistic Regression:" ,accuracy_score(y_test,predict_rf))


# In[290]:


accuracy_rfc = model_rf.score(X_test,y_test)
print(f"Random Forest Classifier Accuracy: {accuracy_rfc}")


# <code style="background:black;color:cyan"> **Original mail texts accuracy :** </code>

# In[291]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['Mails'])

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[292]:


rf= RandomForestClassifier(random_state=42)
model_rf = rf.fit(X_train, y_train)


# In[293]:


accuracy_rfc_org = model_rf.score(X_test,y_test)
print(f"Random Forest Classifier Accuracy: {accuracy_rfc_org}")


# ## <span style="font-family:candara">Executing a Train-Test Split,Applying <mark>Naive Bayes</mark>, and Evaluating Accuracy</span>

# <code style="background:yellow;color:black"> **Corected texts accuracy :** </code>

# In[294]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['corrected_texts']).toarray()

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[295]:


nb= MultinomialNB()
model_nb = nb.fit(X_train, y_train)


# In[296]:


accuracy_nb = model_nb.score(X_test,y_test)
print(f"Naive Bayes Accuracy Score: {accuracy_nb}")


# <code style="background:black;color:cyan"> **Original mail texts accuracy :** </code>

# In[297]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['Mails'])

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[298]:


nb= MultinomialNB()
model_nb = nb.fit(X_train, y_train)


# In[299]:


y_pred_nb = nb.predict(X_test)


# In[300]:


accuracy_nb_org = model_nb.score(X_test,y_test)
print(f"Naive Bayes Accuracy Score: {accuracy_nb_org}")


# ## <span style="font-family:candara">Executing a Train-Test Split,Applying <mark>SVM</mark> model, and Evaluating Accuracy</span>

# <code style="background:yellow;color:black"> **Corected texts accuracy :** </code>

# In[301]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['corrected_texts']).toarray()

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[302]:


svm = SVC(random_state=42)
model_svm = svm.fit(X_train, y_train)
model_svm


# In[303]:


accuracy_svm = model_svm.score(X_test,y_test)
print(f"SVM Accuracy score:", accuracy_svm)


# <code style="background:black;color:cyan"> **Original mail texts accuracy :** </code>

# In[304]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['Mails'])

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[305]:


svm = SVC(random_state=42)
model_svm = svm.fit(X_train, y_train)
model_svm


# In[306]:


y_pred_svm = svm.predict(X_test)


# In[307]:


accuracy_svm_org = model_svm.score(X_test,y_test)
print(f"SVM Accuracy score:", accuracy_svm_org)


# ## <span style="font-family:candara">Executing a Train-Test Split,Applying <mark>Gradient Boosting Classifier</mark> model, and Evaluating Accuracy</span>

# <code style="background:yellow;color:black"> **Corected texts accuracy :** </code>

# In[308]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['corrected_texts']).toarray()

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[309]:


gb = GradientBoostingClassifier(random_state=42)
model_gb = gb.fit(X_train, y_train)

model_gb


# In[310]:


accuracy_gb = model_gb.score(X_test,y_test)
print(f"Gradient boosting Classifier Accuracy score:", accuracy_gb)


# <code style="background:black;color:cyan"> **Original mail texts accuracy :** </code>

# In[311]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['Mails'])

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[312]:


gb = GradientBoostingClassifier(random_state=42)
model_gb = gb.fit(X_train, y_train)


# In[313]:


accuracy_gb_org = model_gb.score(X_test,y_test)
print(f"Gradient boosting Classifier Accuracy score:", accuracy_gb_org)


# ## <span style="font-family:candara">Executing a Train-Test Split,Applying <mark>XGB Classifier</mark> model, and Evaluating Accuracy</span>

# <code style="background:yellow;color:black"> **Corected texts accuracy :** </code>

# In[314]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['corrected_texts']).toarray()

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[315]:


import xgboost as xgb


# In[316]:


xgb= xgb.XGBClassifier(random_state=42)

y_train_binary = y_train.apply(lambda label: 1 if label == 'spam' else 0)
y_test_binary = y_test.apply(lambda label: 1 if label == 'spam' else 0) 

model_xgb = xgb.fit(X_train, y_train_binary)


# In[317]:


accuracy_xgb = model_xgb.score(X_test,y_test_binary)
print(f"XGB Classifier Accuracy score:", accuracy_xgb)


# <code style="background:black;color:cyan"> **Original mail texts accuracy :** </code>

# In[318]:


cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(df['Mails'])

y = df['Mail_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[319]:


import xgboost as xgb


# In[320]:


xgb= xgb.XGBClassifier(random_state=42)

y_train_binary = y_train.apply(lambda label: 1 if label == 'spam' else 0)
y_test_binary = y_test.apply(lambda label: 1 if label == 'spam' else 0) 

model_xgb = xgb.fit(X_train, y_train_binary)


# In[321]:


accuracy_xgb_org = model_xgb.score(X_test,y_test_binary)
print(f"XGB Classifier Accuracy score:", accuracy_xgb_org)


# ## <span style="color:#FF5733;"> Comparing the Regression models Accuracy Score to Finalize the correct Model</span>

# ### <mark> Accuracy score for X variable as Corrected Texts using NLP </mark>

# In[322]:


print("{:<25} {:<10}".format('Algorithm', 'Accuracy'))
print("-----------------------------------------------")

algorithm_names = ['Naive Bayes', 'Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'XGBoost']
accuracies = [accuracy_nb, accuracy_rfc, accuracy_svm, accuracy_lor, accuracy_dtc, accuracy_gb, accuracy_xgb]

for name, accuracy in zip(algorithm_names, accuracies):
    print("{:<25} {:<10.2f}%".format(name, accuracy * 100))


# ### <mark> Accuracy score for X variable as Normal texts recieved in the mail </mark>

# In[323]:


print("{:<25} {:<10}".format('Algorithm', 'Accuracy'))
print("-----------------------------------------------")

algorithm_names = ['Naive Bayes', 'Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'XGBoost']
accuracies1= [accuracy_nb_org, accuracy_rfc_org, accuracy_svm_org, accuracy_lor_org, accuracy_dtc_org, accuracy_gb_org, accuracy_xgb_org]

for name, accuracy in zip(algorithm_names, accuracies1):
    print("{:<25} {:<10.2f}%".format(name, accuracy * 100))


# In[324]:


algorithm_names = ['Naive Bayes', 'Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'XGBoost']
accuracies = [accuracy_nb, accuracy_rfc, accuracy_svm, accuracy_lor, accuracy_dtc, accuracy_gb, accuracy_xgb]

fig = go.Figure()

fig.add_trace(go.Scatter(x=algorithm_names, y=accuracies, mode='lines+markers'))

fig.update_layout(
    title='Comparison of Machine Learning Algorithms used for Accuracy prediction in Corrected texts',
    xaxis_title='Algorithm',
    yaxis_title='Accuracy',
    xaxis_tickangle=35,
    margin=dict(l=0, r=0, b=0, t=40),
)

fig.show()


# In[325]:


algorithm_names = ['Naive Bayes', 'Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'XGBoost']
accuracies1 = [accuracy_nb_org, accuracy_rfc_org, accuracy_svm_org, accuracy_lor_org, accuracy_dtc_org, accuracy_gb_org, accuracy_xgb_org]

fig = go.Figure()

fig.add_trace(go.Scatter(x=algorithm_names, y=accuracies1, mode='lines+markers'))

fig.update_layout(
    title='Comparison of Machine Learning Algorithms used for Accuracy prediction in Normal texts Recieved',
    xaxis_title='Algorithm',
    yaxis_title='Accuracy',
    xaxis_tickangle=30,
    margin=dict(l=0, r=0, b=0, t=40),
)

fig.show()


# ###  From this analysis we can come with the conclude that corrected texts accuracy is slightly higher than normal texts.Naive Bayes, SVM and XGBoost classifiers are performing consistently well with high accuracy (above 97%).Logistic Regression model performing more accgurately than the others with (98% and above)

# ## <span style="color:#FF5733;"> Performance of the model with high accuracy</span>

# In[326]:


ytest = np.array(y_test)
print(classification_report(model_lr.predict(X_test),ytest))
conf_matrix=confusion_matrix(model_lr.predict(X_test),ytest)
print(confusion_matrix(model_lr.predict(X_test),ytest))


# ## <span style="color:#FF5733;"> Confussion matrix </span>

# In[327]:


class_names = ['ham','spam']
conf_matrix = confusion_matrix(model_lr.predict(X_test),ytest)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='OrRd', cbar=True,
            xticklabels=class_names, yticklabels=class_names)

plt.figtext(0.5, 0.95, 'Confusion Matrix', ha='center')

plt.ylabel('Actual',labelpad=50)
plt.xlabel('Predicted',labelpad = 50)

plt.show()

