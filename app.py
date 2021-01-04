from flask import Flask, render_template, url_for, request, jsonify 
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
 

app = Flask(__name__)


reddit = load_model('save_model.h5')




@app.route("/")

def home():
    
    return render_template('reddit_index.html')  


    
@app.route("/predict",methods = ['POST'])
def predict():
    
    
    user_input = request.form['text']
    data = [user_input]
    
    reddit_predict = reddit.predict_proba(data)[:,1]
    reddit_predict1=round(reddit_predict[0],2)
    return render_template('reddit_index.html', reddit_predict= 'prob(upvotes):{}'.format(reddit_predict1))


if __name__ == "__main__":
    print('Hi Main !!')
    app.run(debug=True, threaded=False)
    