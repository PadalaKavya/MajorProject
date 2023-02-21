from flask import Flask,render_template,request,url_for,session,redirect
import pytesseract
from PIL import Image
from flask_mysqldb import MySQL
import MySQLdb.cursors
app = Flask(__name__)
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#connect to Xammp database
app.secret_key = 'a'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'kavya1'
mysql = MySQL(app)


#This is the homepage
@app.route('/')
def home():
    loggedin = getLoginDetails()
    return render_template('Homepage.html',loggedin=loggedin)


#This is registration page for a user
@app.route('/Register',methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM register WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            print("user already exist")
            return render_template('Login.html')
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO register VALUES(NULL,% s,% s,% s)',(name,email,password))
        mysql.connection.commit()
        return redirect(url_for('login'))
    return render_template('Register.html')


#This is to register doctor for doctor
@app.route('/RegisterDoc',methods=['GET','POST'])
def RegisterDoc():
    if request.method == "POST":
        doc_name = request.form["doc_name"]
        qualification = request.form["qualification"]
        doc_email = request.form["doc_email"]
        experience = request.form["experience"]
        edu_backgroud = request.form["edu_background"]
        phone = request.form["phone"]
        password = request.form["password"]
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM docreg WHERE doc_email = % s', (doc_email, ))
        doc_account = cursor.fetchone()
        if doc_account:
            print("user already exist")
            return render_template('doc_Login.html')
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO docreg VALUES(NULL,% s,% s,% s,% s,% s,% s,% s)',(doc_name,qualification,doc_email,experience,edu_backgroud,phone,password))
        mysql.connection.commit()
        return redirect(url_for('login'))
    return render_template('Regdoc.html')


#This is the login page for user
@app.route('/Login',methods=['GET','POST'])
def login():
    global user_id
    msg=' '
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password'] 
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM register WHERE email = % s AND password != % s', (email,password,))
        account2 = cursor.fetchone()
        cursor.execute('SELECT * FROM register WHERE email = % s AND password = % s', (email,password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True 
            session['user_id'] = account['user_id']
            id = account['user_id']
            session['email'] = account["email"]
            return redirect(url_for('Prediction'))
        elif account2:
            msg = "wrong password"
            return redirect(url_for('login'))
        else:
            return redirect(url_for('register'))
    
        #return render_template('Homepage.html')
    return render_template('Login.html') 

#This is the login page for doctor
@app.route('/LoginDoc',methods=['GET','POST'])
def LoginDoc():
    global doc_id
    msg=' '
    if request.method == 'POST' and 'doc_email' in request.form and 'password' in request.form:
        doc_email = request.form['doc_email']
        password = request.form['password'] 
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM docreg WHERE doc_email = % s AND password != % s', (doc_email,password,))
        doc_account2 = cursor.fetchone()
        cursor.execute('SELECT * FROM docreg WHERE doc_email = % s AND password = % s', (doc_email,password,))
        doc_account = cursor.fetchone()
        if doc_account:
            session['loggedin'] = True 
            session['doc_id'] = doc_account['doc_id']
            id = doc_account['doc_id']
            session['doc_email'] = doc_account["doc_email"]
            return redirect(url_for('Prediction'))
        elif doc_account2:
            msg = "wrong password"
            return redirect(url_for('LoginDoc'))
        else:
            return redirect(url_for('RegisterDoc'))
    return render_template('Doc_Login.html') 

#To upload the image as a data
# @app.route('/upload',methods=['GET', 'POST'])
# def upload():
#     imagefile = request.files.get('imagefile', '') 
#     print('imagefile',imagefile)
#     img = Image.open(imagefile)
#     text = pytesseract.image_to_string(img)
#     return 'text'
#sc = pickle.load(open('sc.pkl', 'rb'))

# class BiLSTM_Sentiment_Classifier(nn.Module):

#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, lstm_layers, bidirectional,batch_size, dropout):
#         super(BiLSTM_Sentiment_Classifier,self).__init__()
        
#         self.lstm_layers = lstm_layers
#         self.num_directions = 2 if bidirectional else 1
#         self.hidden_dim = hidden_dim
#         self.num_classes = num_classes
#         self.batch_size = batch_size
        

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
#         self.lstm = nn.LSTM(embedding_dim,
#                             hidden_dim,
#                             num_layers=lstm_layers,
#                             dropout=dropout,
#                             bidirectional=bidirectional,
#                             batch_first=True)

#         self.fc = nn.Linear(hidden_dim*self.num_directions, num_classes)
#         self.softmax = nn.LogSoftmax(dim=1)
        
#     def forward(self, x, hidden):
#         self.batch_size = x.size(0)
#         ##EMBEDDING LAYER
#         embedded = self.embedding(x)
#         #LSTM LAYERS
#         out, hidden = self.lstm(embedded, hidden)
#         #Extract only the hidden state from the last LSTM cell
#         out = out[:,-1,:]
#         #FULLY CONNECTED LAYERS
#         out = self.fc(out)
#         out = self.softmax(out)

#         return out, hidden

#     def init_hidden(self, batch_size):
#         #Initialization of the LSTM hidden and cell states
#         h0 = torch.zeros((self.lstm_layers*self.num_directions, batch_size, self.hidden_dim)).detach().to(DEVICE)
#         c0 = torch.zeros((self.lstm_layers*self.num_directions, batch_size, self.hidden_dim)).detach().to(DEVICE)
#         hidden = (h0, c0)
#         return hidden
# #To make the predictions
# NUM_CLASSES = 5 #We are dealing with a multiclass classification of 5 classes
# HIDDEN_DIM = 100 #number of neurons of the internal state (internal neural network in the LSTM)
# LSTM_LAYERS = 1 #Number of stacked LSTM layers
# BATCH_SIZE = 32
# LR = 3e-4 #Learning rate
# DROPOUT = 0.5 #LSTM Dropout
# BIDIRECTIONAL = True #Boolean value to choose if to use a bidirectional LSTM or not
# EPOCHS = 5 #Number of training epoch
# VOCAB_SIZE=33009
# EMBEDDING_DIM = 200
# #hidden = torch.zeros(num_layers * num_directions, batch, hidden_size)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = BiLSTM_Sentiment_Classifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM,NUM_CLASSES, LSTM_LAYERS,BIDIRECTIONAL, BATCH_SIZE, DROPOUT)
# model = model.to(DEVICE)
# model.load_state_dict(torch.load('./state_dict.pt'))
# model.eval()
# hidden = torch.zeros(1 * 2,32, 2)
# #test_h = model.init_hidden(labels.size(0))
# print("Loaded model")
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/Prediction',methods=['GET','POST'])
def Prediction():
    loggedin = getLoginDetails()
    print(loggedin)
    if loggedin == False:
        return redirect(url_for('login'))
    if request.method == 'POST':
        tweet = request.form.get("tweet")
        tweet = [tweet,]
        print(tweet)
        prediction = model.predict(tweet)
        print(prediction)
        #prediction = model.predict(tweet)
        if(prediction == 0):
            value = 'age'
        elif(prediction == 1):
            value="ethnicity"
        elif(prediction == 2):
            value = 'gender'
        elif(prediction == 3):
            value = "not cyberbullying"
        elif(prediction == 4):
            value = "other cyberbullying"
        elif(prediction == 5):
            value = "religion"
        print(value)
        return render_template('Prediction.html',tweet=tweet,pred=value)
    return render_template('Prediction.html',loggedin=loggedin)



@app.route('/Stories',methods=['GET','POST'])
def Stories():
    loggedin = getLoginDetails()
    if loggedin == False:
        return redirect(url_for('login'))
    return render_template('Stories.html',loggedin=loggedin)


# route to belief story 
@app.route('/Belief',methods=['GET','POST'])
def Belief():
    loggedin = getLoginDetails()
    if loggedin == False:
        return redirect(url_for('login'))
    type = "Belief"
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM stories WHERE type = % s",(type,))
    data = cur.fetchall()
    if request.method == "POST":
        type = "Belief"
        story = request.form["story"]
        title = request.form["title"]
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO stories VALUES(NULL,% s,% s,% s,% s)',(session['user_id'],type,story,title))
        mysql.connection.commit()
        return redirect(url_for('Belief'))
    return render_template("Belief.html",data=data,loggedin=loggedin)

#This is route for age
@app.route('/Age',methods=['GET','POST'])
def Age():
    loggedin = getLoginDetails()
    if loggedin == False:
        return redirect(url_for('login'))
    type = "Age"
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM stories WHERE type = % s",(type,))
    data = cur.fetchall()
    if request.method == "POST":
        type = "Age"
        story = request.form["story"]
        title = request.form["title"]
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO stories VALUES(NULL,% s,% s,% s,% s)',(session['user_id'],type,story,title))
        mysql.connection.commit()
        return redirect(url_for('Age'))
    return render_template("Age.html", data=data,loggedin=loggedin)

#This is route for gender
@app.route('/Gender',methods=['GET','POST'])
def Gender():
    loggedin = getLoginDetails()
    if loggedin == False:
        return redirect(url_for('login'))
    type = "Gender"
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM stories WHERE type = % s",(type,))
    data = cur.fetchall()
    if request.method == "POST":
        type = "Gender"
        story = request.form["story"]
        title = request.form["title"]
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO stories VALUES(NULL,% s,% s,% s,% s)',(session['user_id'],type,story,title))
        mysql.connection.commit()
        return redirect(url_for('Gender'))
    return render_template("Gender.html",data=data,loggedin=loggedin)

#This is route for ethinicity 
@app.route('/Ethnicity',methods=['GET','POST'])
def Ethnicity():
    loggedin = getLoginDetails()
    if loggedin == False:
        return redirect(url_for('login'))
    type = "Ethnicity"
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM stories WHERE type = % s",(type,))
    data = cur.fetchall()
    if request.method == "POST":
        type = "Ethnicity"
        story = request.form["story"]
        title = request.form["title"]
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO stories VALUES(NULL,% s,% s,% s,% s)',(session['user_id'],type,story,title))
        mysql.connection.commit()
        return redirect(url_for('Ethnicity'))
    return render_template("Ethnicity.html", data=data,loggedin=loggedin)

@app.route('/Consultation',methods=['GET','POST'])
def Consultation():
    loggedin = getLoginDetails()
    if loggedin == False:
        return redirect(url_for('login'))
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM docreg")
    data = cur.fetchall()
    return render_template("Consultation.html",data=data,loggedin=loggedin)

def getLoginDetails():
    cursor = mysql.connection.cursor()
    if 'email' not in session:
        if 'doc_email' not in session:
            loggedin = False
            name = ''
            doc_name = ''
        else:
            loggedin = True
    else:
        loggedin = True
    return (loggedin)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('user_id', None)
    session.pop('doc_id', None)
    session.pop('email', None)
    session.pop('doc_email',None)
    return redirect(url_for('home'))
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug = True,port = 8080)


# 6. Logout