import numpy as np 
import pandas as pd 
import transformers
import webbrowser
from threading import Timer
transformers.logging.set_verbosity_error()

from preprocess_data import preprocess_data
from model import create_model
from predict_output import predict
from get_metric import get_metric
from tokenizer import get_tokenizer

from flask import Flask,request,jsonify, render_template

app=Flask(__name__)

def get_predictions(input):
  print(input)
  tokenizer = get_tokenizer('./archive (2)/roberta_tokenizer')

  input_ids,attention_mask,input = preprocess_data(input,128,tokenizer)
  
  model = create_model(128,0.1,'./archive (1)/mymodel_pretrained','./archive/checkpt1/model_roberta.hdf5')

  input_data = (input_ids,attention_mask)
  pred_text= predict(model,input_data,tokenizer,input)
  return pred_text

print('Hello World')
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/result',methods=['POST'])	
def result():
	vals  = [each for each in request.form.values()]
	values = {'text':[vals[0]],'sentiment':[vals[1]]}
	df = pd.DataFrame(values)
	output =get_predictions(df)
	return render_template('final.html',text=vals[0],prediction=output[0],sentiment=vals[1])




def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')


if __name__ =="__main__":
    #app.run(debug=True)
#if __name__ == "__main__":
      Timer(1, open_browser).start();
      app.run(port=5000)  	