#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, flash, jsonify, redirect, url_for, session, request, logging
from passlib.hash import sha256_crypt
from flask_mysqldb import MySQL
from sqlhelpers import *
from forms import * 
from functools import wraps
from final_model import *
import tensorflow as tf
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import string
import re
import joblib
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
import keras.api._v2.keras as keras
import tensorflow.keras
#
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense, Flatten, Conv1D, MaxPooling1D, SimpleRNN, GRU, LSTM, LSTM, Input, Embedding, TimeDistributed, Flatten, Dropout,Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import nltk
import json
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

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



@app.route("/" ,methods=['GET'])
def index():    
    return render_template('index.html')


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
            return redirect(url_for('dashboard'))
        else:
            flash('User already exists', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html', form=form)


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
            flash("Username is not found", 'danger')
            return redirect(url_for('login'))
        else:
            #verify that the password entered matches the actual password
            if sha256_crypt.verify(candidate, accPass):
                #log in the user and redirect to Dashboard page
                log_in_user(username)
                flash('You are now logged in.', 'success')
                return redirect(url_for('dashboard'))
            else:
                #if the passwords do not match
                flash("Invalid password", 'danger')
                return redirect(url_for('login'))

    return render_template('login.html')



@app.route("/logout")
@is_logged_in
def logout():
    session.clear()
    flash("Logout success", "success")
    return redirect(url_for('login'))





@app.route("/dashboard")
@is_logged_in
def dashboard():
    return render_template('dashboard.html', session=session,page='dashboard')

@app.route("/quiz1")
def quiz1():    
    return render_template('quiz1.html')

@app.route("/quiz2")
def quiz2():    
    return render_template('quiz2.html')

@app.route("/quiz3")
def quiz3():    
    return render_template('quiz3.html')

@app.route("/quiz4")
def quiz4():    
    return render_template('quiz4.html')

@app.route("/quiz5")
def quiz5():    
    return render_template('quiz5.html')

@app.route("/questionnaire")
def questionnaire():    
    return render_template('quizDashboard.html')

@app.route("/solution1")
def solution1():    
    return render_template('solution1.html')

@app.route("/solution2")
def solution2():    
    return render_template('solution2.html')

@app.route("/solution3")
def solution3():    
    return render_template('solution3.html')

@app.route("/solution4")
def solution4():    
    return render_template('solution4.html')

@app.route("/solution5")
def solution5():    
    return render_template('solution5.html')

@app.route("/solutiondash")
def solutiondashboard():    
    return render_template('solutionDashboard.html')

@app.route("/about")
def about():    
    return render_template('about.html')

#----------------------------------------------------------------------------------------------
# @app.route('/predict', methods=['POST'])
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
#     return response

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    response = None
    if request.method == 'POST':
        df_input = request.form['question']
        df_input = get_text(df_input)
        tokenizer_t = joblib.load('tokenizer_t.pkl')
        vocab = joblib.load('vocab.pkl')
        df_input = remove_stop_words_for_input(tokenizer,df_input,'questions')
        encoded_input = encode_input_text(tokenizer_t,df_input,'questions')
        pred = get_pred(model1,encoded_input)
        pred = bot_precausion(df_input,pred)
        response = get_response(df2,pred)
    return render_template('chatbot.html', response=response)


if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.run(debug=True)   
    #app.run()