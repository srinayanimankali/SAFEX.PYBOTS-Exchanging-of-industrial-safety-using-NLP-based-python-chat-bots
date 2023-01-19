#!/usr/bin/env python
# coding: utf-8

# ## Domain: Industrial Safety Support - NLP based ChatBot - Safex Pybot Version 3.0 
# 
# #### Context : 
# 
# **** Great Learning Company is looking forward to design an automation which can interact with the user, understand the problem      and display the resolution procedure 
# 
# **** The company is looking for a designed chatbot utility which can help the professionals to highlight the safety risk as per      the incident description usign a ML or DL or redirect the request to an actual human support executive if the request is        complex or not in its database.
# 
# #### Data Description : 
# 
# **** The corpus is attached for the reference. Please enhance/add more data to the corpus using your linguistics skills.

# In[1]:


#Importing the data and files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats; from scipy.stats import zscore, norm, randint
import pickle
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# import the libraries

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow

import tensorflow as tf
import tflearn
import random


# In[3]:


import nltk
nltk.download('punkt')


# In[5]:


use_pretrained_model=True


# ### Modeling and functions for Question and Answers

# In[6]:


# import our chat-bot intents file
import json
with open('safexpybot_intents_qna.json') as json_data:
    intents = json.load(json_data)
    


# In[7]:


# Initialize data fields for our file

words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))


# In[8]:


# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
X_train = list(training[:,0])
y_train = list(training[:,1])


# In[10]:


# reset underlying graph data
from tensorflow.python.framework import ops
ops.reset_default_graph()
    
    # Build neural network
net_qna = tflearn.input_data(shape=[None, len(X_train[0])])
net_qna = tflearn.fully_connected(net_qna, 16)
net_qna = tflearn.fully_connected(net_qna, 16)
net_qna = tflearn.fully_connected(net_qna, len(y_train[0]), activation='softmax')
net_qna = tflearn.regression(net_qna)

# Define model and setup tensorboard
model_qna = tflearn.DNN(net_qna, tensorboard_dir='tflearn_logs')

if use_pretrained_model==False:
    # Start training (apply gradient descent algorithm)
    model_qna.fit(X_train, y_train, n_epoch=1000, batch_size=8, show_metric=True)
    model_qna.save('model_qna.tflearn')
    
    pickle.dump( {'words':words, 'classes':classes, 'X_train':X_train, 'y_train':y_train}, open( "qna_training_data", "wb" ) )


# In[11]:


# restore all of our data structures
import pickle
data_qna = pickle.load( open( "qna_training_data", "rb" ) )
words_qna = data_qna['words']
classes_qna = data_qna['classes']
X_train_qna = data_qna['X_train']
y_train_qna = data_qna['y_train']

# import our chat-bot intents file
import json
with open('safexpybot_intents_qna.json') as json_data:
    intents_qna = json.load(json_data)


# In[12]:


# load our saved model
#if use_pretrained_model==True:
model_qna.load('./model_qna.tflearn')


# In[13]:


# functions for cleansing the sentenses and to get the bag of words

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


# In[14]:


# functions to classify the sentences and for respective responses
# create a data structure to hold user context

context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model_qna.predict([bow(sentence, words_qna)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes_qna[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def response(sentence, show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents_qna['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    #return print('BOT: ', random.choice(i['responses']))
                    result=random.choice(i['responses'])
                    return result

            results.pop(0)
            


# ## Chatbot : Historical Data Insights

# In[18]:


dataset= pd.read_csv('data.csv')
#print("Shape of the dataset is :",dataset.shape)
#dataset.head()


# Creating a list of possible features allowed for showing the data insights

# In[19]:


dataset_features=['Country', 'Local', 'Industry Sector', 'Accident Level',
       'Potential Accident Level', 'Gender', 'Natureofemployee','Year','Local','Season']


# ### function to be called by the Chatbot for showing data insights

# In[20]:


def show_eda(response_result):
    eda_flag = True
    result = list(response_result.split(","))
    result_items = len(result)
    
    for item in result:
        if item not in dataset_features:
            eda_flag = False
    
    if eda_flag == True:
        if result_items == 1:
            univariate_analysis_categorical(dataset,result[0])
        elif result_items == 2:
           
            bivariate_analysis_categorical(dataset,result[0],result[1])
        else:
            print("Sorry, not able to find appropriate answer. Please ask teh question again")
    else:
        print(response_result)  
        
        
            
    


# In[21]:


def univariate_analysis_categorical(dataset,feature):    
    print("\n")
    print("===========================================================================================")
    print("Data Analysis of feature: ",feature)
    print("===========================================================================================\n")
                
    print("\n")  
    print("-----------------")
    print("Countplot  for feature: ",feature)
    print("-----------------")
        
    plt.figure(figsize=(10,6))
    sns.countplot(dataset[feature],order = dataset[feature].value_counts().index)
    plt.xticks(rotation = 'vertical')
    plt.show()
    
    print("-----------------")
    print("Pie Chart for feature: ",feature)
    print("------------------")      
        
    labels=dataset[feature].unique()
    plt.figure(figsize=(10,6))
    dataset[feature].value_counts().plot.pie(autopct="%.1f%%")
    plt.show()
            
    
    print("\n")
    print("-----------------")
    print("Value Counts for feature: ",feature)
    print("-------------------")
    
    print(dataset[feature].value_counts().sort_values(ascending=False))
    print('')


# In[22]:


def bivariate_analysis_categorical(dataset,feature1,feature2):
    
       
            if feature1 != feature2:              
                   

                print("\n")
                print("===========================================================================================")
                print("Data Analysis of features: ",feature1,' and  ', feature2)
                print("===========================================================================================")

                
                bivariate_analysis_df = pd.crosstab(index=dataset[feature1], 
                                          columns=dataset[feature2])
                
                print("\n")
                print("------------------------------------------")
                print("Cross table Analysis of features: ",feature1,' and  ', feature2)
                print("------------------------------------------")
                
                #display(bivariate_analysis_df)
                
                print("\n")
                print("------------------------------------------")
                print("Count plot Analysis of features: ",feature1,' and  ', feature2)
                print("------------------------------------------")
                
                plt.figure(figsize=(12,6))
                sns.countplot(x=feature1, hue=feature2, data=dataset)
                plt.show()
                


# ## Modeling and Functions for Historical Data Insights through chatbot

# In[23]:


# import our chat-bot intents file
import json
with open('safexpybot_intents_eda.json') as json_data:
    intents_eda = json.load(json_data)

# Initialize data fields for our file

words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents_eda['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

#print (len(documents), "documents")
#print (len(classes), "classes", classes)
#print (len(words), "unique stemmed words", words)


# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
X_train = list(training[:,0])
y_train = list(training[:,1])


# reset underlying graph data
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Build neural network
net_eda = tflearn.input_data(shape=[None, len(X_train[0])])
net_eda = tflearn.fully_connected(net_eda, 16)
net_eda = tflearn.fully_connected(net_eda, 16)
net_eda = tflearn.fully_connected(net_eda, len(y_train[0]), activation='softmax')
net_eda = tflearn.regression(net_eda)

# Define model and setup tensorboard
model_eda = tflearn.DNN(net_eda, tensorboard_dir='tflearn_logs')

if use_pretrained_model==False:
    # Start training (apply gradient descent algorithm)
    model_eda.fit(X_train, y_train, n_epoch=1000, batch_size=8, show_metric=True)
    model_eda.save('model_eda.tflearn')
    pickle.dump( {'words':words, 'classes':classes, 'X_train':X_train, 'y_train':y_train}, open( "eda_training_data", "wb" ) )


# In[24]:


# restore all of our data structures
import pickle
data_eda = pickle.load( open( "eda_training_data", "rb" ) )
words_eda = data_eda['words']
classes_eda = data_eda['classes']
X_train_eda = data_eda['X_train']
y_train_eda = data_eda['y_train']


# In[25]:


# import our chat-bot intents file
import json
with open('safexpybot_intents_eda.json') as json_data:
    intents_eda = json.load(json_data)

#if use_pretrained_model==True:
# load our saved model
model_eda.load('./model_eda.tflearn')


# In[26]:



# functions to classify the sentences and for respective responses
# create a data structure to hold user context

context = {}

ERROR_THRESHOLD = 0.25
def classify_eda(sentence):
    # generate probabilities from the model
    results = model_eda.predict([bow(sentence, words_eda)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes_eda[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def response_eda(sentence, show_details=False):
    
    results = classify_eda(sentence)
    
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents_eda['intents']:
                
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    
                    #return print(random.choice(i['responses']))
                    response_result = random.choice(i['responses'])
                    show_eda(response_result)
                    

            results.pop(0)
            


# In[27]:


#print(response_eda('analysis for Accident'))


# ## Chatbot : Classification Model

# In[28]:


dataset_clf = dataset.copy()


# In[29]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
#from wordcloud import WordCloud, STOPWORDS
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import string
nltk.download('stopwords')
stop=set(stopwords.words('english'))

def preprocess_text(text):
      corpus=[]
      #stem=PorterStemmer()
      lem=WordNetLemmatizer()
      for news in text:
          words=[w for w in word_tokenize(news) if (w not in stop)]
          
          words=[lem.lemmatize(w) for w in words if len(w)>2]
          words = [''.join(c for c in s if c not in string.punctuation) for s in words if s]
          words = [word.lower() for word in words]
          words = [word for word in words if word.isalpha()]
          corpus.append(words) 
         
      return corpus      
      
dataset_clf['processed_text']= preprocess_text(dataset_clf['Description'])

desc_processed = []
for i in range(len(dataset_clf['processed_text'])):
   desc_processed.append(' '.join(wrd for wrd in dataset_clf.iloc[:,14][i]))

dataset_clf['description_processed'] = desc_processed


# ### Modeling and functions for  Accident Level

# In[31]:


#Count vectorization
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

X = dataset_clf['description_processed']
y = dataset_clf['Accident Level']

vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
Xc = vectorizer.fit_transform(X).toarray()
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, y, test_size=0.15, random_state=42)


# In[34]:


#SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC

svc = LinearSVC( max_iter=2500)
if use_pretrained_model==False:
    svc.fit(Xc_train, yc_train)
    yc_pred_SVC = svc.predict(Xc_test)
    acc_svc = accuracy_score(yc_test, yc_pred_SVC)
    acc_svc_tr = svc.score(Xc_train, yc_train)
    print('\n')
    print("Train accuracy of the SVC model : {:.2f}".format(acc_svc_tr*100))
    print('\n')
    print("Test accuracy of the SVC model : {:.2f}".format(acc_svc*100))
    print('\n')
    print('Classification report:',classification_report(yc_test, yc_pred_SVC))
    # save the model
    filename = 'svc_model_acclevel.sav'
    pickle.dump(svc, open(filename, 'wb'))


# In[35]:


loaded_model_acclevel = pickle.load(open('./svc_model_acclevel.sav', 'rb'))


# In[36]:


def response_acclevel(sentence):
    
    df_user_input=pd.DataFrame()
    df=pd.DataFrame()
    user_input = {'Description': sentence}
    df_user_input = df_user_input.append(user_input, ignore_index = True)
    df_user_input['processed_text']= preprocess_text(df_user_input['Description'])
    df['processed_text']=df_user_input['processed_text']
    df.reset_index(inplace=True)
    desc_processed = []
    for i in range(len(df['processed_text'])):
        desc_processed.append(' '.join(wrd for wrd in df.iloc[:,1][i]))
    test = vectorizer.transform(desc_processed)
    result = loaded_model_acclevel.predict(test)
    return result
    


# ### Modeling and functions for Potential Accident Level

# In[37]:


#Count vectorization
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

X = dataset_clf['description_processed']
y = dataset_clf['Potential Accident Level']

vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
Xc = vectorizer.fit_transform(X).toarray()
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, y, test_size=0.15, random_state=42)


# In[38]:


#SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC

svc_potacclevel = LinearSVC( max_iter=2500)
if use_pretrained_model==False:
    svc_potacclevel.fit(Xc_train, yc_train)
    yc_pred_SVC_potacclevel = svc_potacclevel.predict(Xc_test)
    acc_svc_potacclevel = accuracy_score(yc_test, yc_pred_SVC_potacclevel)
    acc_svc_tr_potacclevel = svc.score(Xc_train, yc_train)
    print('\n')
    print("Train accuracy of the SVC model : {:.2f}".format(acc_svc_tr_potacclevel*100))
    print('\n')
    print("Test accuracy of the SVC model : {:.2f}".format(acc_svc_potacclevel*100))
    print('\n')
    print('Classification report:',classification_report(yc_test, yc_pred_SVC_potacclevel))
    #save the model
    filename = 'svc_model_potacclevel.sav'
    pickle.dump(svc_potacclevel, open(filename, 'wb'))


# In[39]:


loaded_model_potacclevel = pickle.load(open('./svc_model_potacclevel.sav', 'rb'))


# In[40]:


def response_potacclevel(sentence):
    
    df_user_input=pd.DataFrame()
    df=pd.DataFrame()
    user_input = {'Description': sentence}
    df_user_input = df_user_input.append(user_input, ignore_index = True)
    df_user_input['processed_text']= preprocess_text(df_user_input['Description'])
    df['processed_text']=df_user_input['processed_text']
    df.reset_index(inplace=True)
    
    desc_processed = []
    for i in range(len(df['processed_text'])):
        desc_processed.append(' '.join(wrd for wrd in df.iloc[:,1][i]))
    test = vectorizer.transform(desc_processed)
    result = loaded_model_potacclevel.predict(test)
    return result
    


# ## Chatbot with Conditions

# ### Chatbot function for Question and Answers

# In[41]:


def chatbot_qna():
    flag=True
    print('\n')
    print("BOT: I am your Virtual assistant. I will try to answer your questions on Accidents Data!     \n \t If you want to exit any time, just type Bye!")
    while(flag==True):
        print('\n')
        user_response = input("You: ")
        user_response=user_response.lower()
        if(user_response!='bye') and (user_response!='exit') and (user_response!='quit'):
            if(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                print('\n')
                print("BOT: You are welcome..")
                print("BOT: Exited from Question and Answer Module")

            else:
                print('\n')
                response(user_response)
                print('\n')
        else:
            flag=False
            print('\n')
            print("BOT: Exited from Question and Answer Module")
            #print("BOT: Goodbye! Take care  ")
            print('\n')
            print('-----------------------------------------------------')
            print('\n')
            


# ### Chatbot function for Accident Level Classification

# In[42]:


def chatbot_acclevel():
    flag=True
    print('\n')
    print("BOT: I am your Virtual assistant. I will try to Predict Accident Level for your description     \n \t If you want to exit any time, just type Bye!")
    while(flag==True):
        print('\n')
        user_response = input("You: ")
        user_response=user_response.lower()
        if(user_response!='bye') and (user_response!='exit') and (user_response!='quit'):
            if(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                print('\n')
                print("BOT: You are welcome..")
                print("BOT: Exited from Accident Level Predictor")

            else:
                print('\n')
                response = response_acclevel(user_response)
                
                if response:
                    print ("Bot: Accident Level is --> ",response[0])
                print('\n')
        else:
            flag=False
            print('\n')
            print("BOT: Exited from Accident Level Predictor Module")
            #print("BOT: Goodbye! Take care  ")
            print('\n')
            print('-----------------------------------------------------')
            print('\n')


# ### Chatbot function for Potential Accident Level Classification

# In[43]:


def chatbot_potacclevel():
    flag=True
    print('\n')
    print("BOT: I am your Virtual assistant. I will try to predict Potential Accident Level for your description     \n \t If you want to exit any time, just type Bye!")
    while(flag==True):
        print('\n')
        user_response = input("You: ")
        user_response=user_response.lower()
        if(user_response!='bye') and (user_response!='exit') and (user_response!='quit'):
            if(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                print('\n')
                print("BOT: You are welcome..")
                print("BOT: Exited from Potential Accident Level Predictor")

            else:
                print('\n')
                response = response_potacclevel(user_response)
                
                if response:
                    print ("Bot: Potential Accident Level is --> ",response[0])
                print('\n')
        else:
            flag=False
            print('\n')
            print("BOT: Exited from Potential Accident Level Predictor Module")
            #print("BOT: Goodbye! Take care  ")
            print('\n')
            print('-----------------------------------------------------')
            print('\n')


# ### Chat bot function for showing Historical Data Insights

# In[44]:


def chatbot_eda():
    flag=True
    print('\n')
    print("BOT: I am your Virtual assistant. I will help you with Insights of Accident Historical Data!           \n \t If you want to exit any time, just type Bye!")
    while(flag==True):
        
        print('\n')
        user_response = input("You: ")
        user_response=user_response.lower()
        if(user_response!='bye') and (user_response!='exit') and (user_response!='quit'):
            if(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                print('\n')
                print("BOT: You are welcome..")
                print("BOT: Exited from Data Insights Module")

            else:                    
                    print('\n')
                    response_eda(user_response)
                    print('\n')
        else:
            flag=False
            print('\n')
            #print("BOT: Goodbye! Take care  ")
            print("BOT: Exited from Data Insights Module")
            print('\n')
            print('-----------------------------------------------------')
            print('\n')


# ## Chatbot : Home Page

# In[45]:

def chatbot_main():
    mainflag=True
    flag = False
    
    while(mainflag==True):
        if flag == False:
            print('')
            print("BOT: Please Select appropriate option:           \n \t            \n \t 1 --> For Question and Answers           \n \t 2 --> For finding the Accident level for your problem           \n \t 3 --> For finding the Potential Accident level for your problem           \n \t 4 --> For Historical Data Insights           \n \t            \n \t if you want to exit any time, just type Bye!")
            
        print('\n')
        user_response = input("You: ")
        user_response=user_response.lower()
        if(user_response!='bye') and (user_response!='exit') and (user_response!='quit'):
            if(user_response=='thanks' or user_response=='thank you' ):
                mainflag=False
                print('\n')
                print("BOT: You are welcome..")
            
            elif (user_response=='1'):
                    chatbot_qna()
            elif (user_response=='2'):
                    chatbot_acclevel()
            elif (user_response=='3'):
                    chatbot_potacclevel()   
            elif (user_response=='4'):
                    chatbot_eda()  
                    
        else:
            mainflag=False
            print('\n')
            print("BOT: Goodbye! Take care  ")
            print('\n')
            print('================================================================')
            print('\n')


# In[ ]:





# In[ ]:




