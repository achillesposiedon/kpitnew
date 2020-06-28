"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from KPIT import app
algos=['Linear Regression','AdaBoost','Decision Tree','Random Forest','SVM','Lasso','Ridge Regression','Catboost','ElasticNet']
@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    algos=['Linear Regression','AdaBoost','Decision Tree','Random Forest','SVM','Lasso','Ridge Regression','Catboost','ElasticNet']
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
        algos=algos
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='V Mohammad Aaftab',
        year=datetime.now().year,
        message='Contact me here.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        
        message='Your application description page.'
    )

@app.route('/diff')
def diff():
    """Renders the about page."""
    algos=['Linear Regression','AdaBoost','Decision Tree','Random Forest','SVM','Lasso','Ridge Regression','Catboost','ElasticNet']
    return render_template(
        'diff.html',
        title='Predict Temperature Difference',
        year=datetime.now().year,
        message='Your application description page.',
        algos=algos
    )

import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

modellnr = pickle.load(open('linearend.pkl', 'rb'))
modelada = pickle.load(open('adaboostend.pkl', 'rb'))
modelcat = pickle.load(open('catboostend.pkl', 'rb'))
modeldec = pickle.load(open('decisiontreeend.pkl', 'rb'))
modelels = pickle.load(open('elasticnetend.pkl', 'rb'))
modelgrad = pickle.load(open('gradientboostend.pkl', 'rb'))
modellaso = pickle.load(open('lassoend.pkl', 'rb'))
modelrand = pickle.load(open('randomforestend.pkl', 'rb'))
modelridge = pickle.load(open('ridgeend.pkl', 'rb'))
modelsvm = pickle.load(open('svmend.pkl', 'rb'))

@app.route('/prediction',methods=['POST'])
def prediction():
    '''
    For rendering results on HTML GUI
    '''
    features = [(x) for x in request.form.values()]
    if features[3]=='Linear Regression':
        model=modellnr
    elif features[3]=='AdaBoost':
        model=modelada
    elif features[3]=='Decision Tree':
        model=modeldec
    elif features[3]=='Random Forest':
        model=modelrand
    elif features[3]=='SVM':
        model=modelsvm
    elif features[3]=='Lasso':
        model=modellaso
    elif features[3]=='Ridge Regression':
        model=modelridge
    elif features[3]=='Catboost':
        model=modelcat
    if features[3]=='ElasticNet':
        model=modelels
    defaultfeatures=[]
    final_features = [np.array(([ 1.00714554e+00,  np.float64(features[1]),  np.float64(features[0]),  2.98146997e+02, 2.94611546e+02,   np.float64(features[2]),  6.29557843e+01,  4.82545513e+01,-1.47012330e+01,  1.22139512e+00,  1.55231290e+00,  1.31437841e+01,2.47449572e+01,  1.96042819e+01,  3.10461801e+03,  1.93399782e+02,2.53854833e-01,  1.12297856e+00, -1.41468786e-01, -1.81799793e-01,-9.61727289e-02,  8.64986837e-03,  8.64986837e-03, -3.14935518e+06,7.73223016e-01,  3.91124483e-02,  2.54606995e+00,  1.58875517e+00]))]
    prediction = model.predict(final_features)

    return render_template('index.html',p='Your model is {}'.format(str(features[3])), prediction_text='End Temperature prediction is {}'.format(np.float64(prediction)),algos=algos)

modellnrd = pickle.load(open('lineardiff.pkl', 'rb'))
modeladad = pickle.load(open('adaboostdiff.pkl', 'rb'))
modelcatd = pickle.load(open('catboostdiff.pkl', 'rb'))
modeldecd = pickle.load(open('decisiontreediff.pkl', 'rb'))
modelelsd = pickle.load(open('elasticnetdiff.pkl', 'rb'))
modelgradd = pickle.load(open('gradientboostdiff.pkl', 'rb'))
modellasod = pickle.load(open('lassodiff.pkl', 'rb'))
modelrandd = pickle.load(open('randomforestdiff.pkl', 'rb'))
modelridged = pickle.load(open('ridgediff.pkl', 'rb'))
modelsvmd = pickle.load(open('svmdiff.pkl', 'rb'))

@app.route('/predictiondiff',methods=['POST'])
def predictiondiff():
    '''
    For rendering results on HTML GUI
    '''
    features = [(x) for x in request.form.values()]
    if features[3]=='Linear Regression':
        model=modellnrd
    elif features[3]=='AdaBoost':
        model=modeladad
    elif features[3]=='Decision Tree':
        model=modeldecd
    elif features[3]=='Random Forest':
        model=modelrandd
    elif features[3]=='SVM':
        model=modelsvmd
    elif features[3]=='Lasso':
        model=modellasod
    elif features[3]=='Ridge Regression':
        model=modelridged
    elif features[3]=='Catboost':
        model=modelcatd
    if features[3]=='ElasticNet':
        model=modelelsd
    defaultfeatures=[]
    final_features = [np.array(([ 1.00714554e+00,  np.float64(features[1]),  np.float64(features[0]),  2.98146997e+02, 2.94611546e+02,   np.float64(features[2]),  6.29557843e+01,  4.82545513e+01,-1.47012330e+01,  1.22139512e+00,  1.55231290e+00,  1.31437841e+01,2.47449572e+01,  1.96042819e+01,  3.10461801e+03,  1.93399782e+02,2.53854833e-01,  1.12297856e+00, -1.41468786e-01, -1.81799793e-01,-9.61727289e-02,  8.64986837e-03,  8.64986837e-03, -3.14935518e+06,7.73223016e-01,  3.91124483e-02,  2.54606995e+00,  1.58875517e+00]))]
    prediction = model.predict(final_features)

    return render_template('diff.html',p='Your model is {}'.format(str(features[3])), prediction_text='Temperature diffference prediction is {}'.format(np.float64(prediction)),algos=algos)

