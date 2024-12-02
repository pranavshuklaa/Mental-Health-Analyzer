#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, flash, jsonify, redirect, url_for, session, request, logging
from passlib.hash import sha256_crypt
from flask_mysqldb import MySQL
from sqlhelpers import *
from forms import * 
from functools import wraps
# from model2 import *

# initialize the app
app = Flask(__name__)


# configure mysql
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'tJVzg22u$5F1'
app.config['MYSQL_DB'] = 'backend'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# initialize mysql
mysql = MySQL(app)
# mysql = MySQL(app)

# import pickle
# from keras.models import load_model

# with open('tokenizer_t.pkl', 'rb') as handle:
#     tokenizer = pickle.load(handle)
# with open('vocab.pkl', 'rb') as handle:
#     vocab = pickle.load(handle)
# model = load_model("model-v1.h5")






#wrap to define if the user is currently logged in from session
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash("Unauthorized, please login.", "danger")
            return redirect(url_for('login'))
    return wrap

def log_in_user(username):
    users = Table("users", "name", "email", "username", "password")
    user = users.getone("username", username)

    session['logged_in'] = True
    session['username'] = username
    session['name'] = user.get('name')
    session['email'] = user.get('email')

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    users = Table("users", "name", "email", "username", "password")

    # if form is submitted
    if request.method == 'POST' and form.validate():
        # collect form data
        username = form.username.data
        email = form.email.data
        name = form.name.data

        # make sure user does not already exist
        if isnewuser(username):
            # add the user to mysql and log them in
            password = sha256_crypt.encrypt(form.password.data)
            users.insert(name, email, username, password)
            log_in_user(username)
            return jsonify({'status': 'success', 'username': username, 'email': email, 'name': name, 'password': password})
        else:
            return jsonify({'status': 'failure', 'message': 'User already exists'})

    return jsonify({'status': 'failure', 'message': 'Invalid form submission'})



@app.route("/login", methods = ['GET', 'POST'])
#@is_logged_in
def login():
    #if form is submitted
    if request.method == 'POST':
        #collect form data
        username = request.form['username']
        candidate = request.form['password']

        #access users table to get the user's actual password
        users = Table("users", "name", "email", "username", "password")
        user = users.getone("username", username)
        accPass = user.get('password')

        #if the password cannot be found, the user does not exist
        if accPass is None:
            return jsonify({'status': 'failure', 'message': 'Username not found'})
        else:
            #verify that the password entered matches the actual password
            if sha256_crypt.verify(candidate, accPass):
                #log in the user
                log_in_user(username)
                return jsonify({'status': 'success', 'message': 'You are now logged in.', 'username': username})
            else:
                #if the passwords do not match
                return jsonify({'status': 'failure', 'message': 'Invalid password'})

    return jsonify({'status': 'failure', 'message': 'Invalid form submission'})


@app.route("/logout")
@is_logged_in
def logout():
    session.clear()
    return jsonify({'status': 'success', 'message': 'Logout success'})


@app.route("/", methods=['GET'])
def index():    
    return jsonify({'status': 'success', 'message': 'Index page'})


@app.route("/dashboard", methods=['GET'])
@is_logged_in
def dashboard():
    return jsonify({'status': 'success', 'message': 'Dashboard page', 'username': session['username']})


# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     return str(model.df_input(userText))

#@app.route('/predict', methods=['POST'])
# def predict():

#     # df_input = request.form['question']
#     # df_input = get_text('question')
#     df_input = request.form['question']
#     df_input = get_text(df_input)
#     # load artifacts 
#     tokenizer_t = joblib.load('tokenizer_t.pkl')
#     vocab = joblib.load('vocab.pkl')

#     df_input = remove_stop_words_for_input(tokenizer,df_input,'questions')
#     encoded_input = encode_input_text(tokenizer_t,df_input,'questions')

#     pred = get_pred(model1,encoded_input)
#     pred = bot_precausion(df_input,pred)

#     response = get_response(df2,pred)
#     return jsonify({'response': response})

# import re
# import random
# from flask import Flask, request, jsonify


# def generate_answer(pattern): 
#     text = []
#     txt = re.sub('[^a-zA-Z\']', ' ', pattern)
#     txt = txt.lower()
#     txt = txt.split()
#     txt = " ".join(txt)
#     text.append(txt)
        
#     x_test = tokenizer.texts_to_sequences(text)
#     x_test = np.array(x_test).squeeze()
#     x_test = pad_sequences([x_test], padding='post', maxlen=X.shape[1])
#     y_pred = model.predict(x_test)
#     y_pred = y_pred.argmax()
#     tag = lbl_enc.inverse_transform([y_pred])[0]
#     responses = df[df['tag'] == tag]['responses'].values[0]

#     return random.choice(responses)

@app.route('/predict', methods=['POST'])
def predict():
    df_input = request.form['question']
    response = generate_answer(df_input)
    return render_template('result.html', response=response)


if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.run(debug=True,port=3000)   
    #app.run()