import mysql.connector
import json
import pickle
from SVD import SVDPredictor
from RecData import RecData
from flask import Flask, jsonify, render_template, Response

SUCCESS_CODE = 200
SUCCESS_EMPTY = 204

with open('model.pkl', 'rb') as file:
    svd, data, train, val, test = pickle.load(file)

app = Flask(__name__, '/templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search.<string:title>')
def search(title):
    return (str(data.search_title(title)[:10]), SUCCESS_CODE)

@app.route('/search.')
def empty_search():
    return ('', SUCCESS_EMPTY)

@app.route('/topn/<user_id>', methods=['GET'])
def get_topn(user_id):
    top_n = svd.top_n(int(user_id))
    return str([data.index_to_title(index) for _, index in top_n])


# Testing routes
@app.route('/insert')
def insert_test():
    mydb = mysql.connector.connect(
        host="mysqldb",
        user="root",
        password="p@ssw0rd1",
        database="inventory"
    )
    cursor = mydb.cursor()
    cursor.execute("INSERT INTO widgets VALUES ('Potato', 'A yummy vegetable.')")
    cursor.close()

    mydb.commit()

    return "insert"

@app.route('/widgets')
def get_widgets():
    mydb = mysql.connector.connect(
        host="mysqldb",
        user="root",
        password="p@ssw0rd1",
        database="inventory"
    )
    cursor = mydb.cursor()


    cursor.execute("SELECT * FROM widgets")

    row_headers=[x[0] for x in cursor.description] #this will extract row headers

    results = cursor.fetchall()
    json_data=[]
    for result in results:
        json_data.append(dict(zip(row_headers,result)))

    cursor.close()

    return json.dumps(json_data)

@app.route('/initdb')
def db_init():
    mydb = mysql.connector.connect(
        host="mysqldb",
        user="root",
        password="p@ssw0rd1"
    )
    cursor = mydb.cursor()

    cursor.execute("DROP DATABASE IF EXISTS inventory")
    cursor.execute("CREATE DATABASE inventory")
    cursor.execute("USE inventory")

    cursor.execute("DROP TABLE IF EXISTS widgets")
    cursor.execute("CREATE TABLE widgets (name VARCHAR(255), description VARCHAR(255))")
    cursor.close()

    return 'init database'

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)