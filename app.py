import mysql.connector
import json
import pickle
import base64
import os

from flask import Flask, jsonify, render_template, request

SUCCESS_CODE = 200
SUCCESS_EMPTY = 204

with open('ens_knn.pkl', 'rb') as file:
    data, ens_knn = pickle.load(file)

app = Flask(__name__, '/templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search.<string:title>')
def search(title):
    return (jsonify(data.search_title(title)[:10]), SUCCESS_CODE)

@app.route('/search.')
def empty_search():
    return (jsonify([]), SUCCESS_EMPTY)

@app.route('/recs', methods=['POST'])
def recs():
    user_data = request.get_json()
    prefs = [(int(index), pref) for index, pref in user_data.items()]
        
    prefs = data.create_prefs(prefs)
    top = ens_knn.top_n(-1, 10, prefs=prefs)
    recs = [(index, data.index_to_id(index), data.index_to_title(index)) for _, index in top]

    return (jsonify(recs), SUCCESS_CODE)

@app.route('/get_cover/<item_id>', methods=['GET'])
def get_cover(item_id):
    with open("static/covers/" + str(item_id) + ".jpg", 'rb') as f:
        img_data = str(base64.b64encode(f.read()).decode('utf-8'))
    return (jsonify(img_data, SUCCESS_CODE))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)