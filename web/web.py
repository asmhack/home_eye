# coding=utf-8
from pprint import pprint

from flask import Flask, render_template, request

from utils import FacesModel

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index_action():
    db = FacesModel.Instance()
    vectors = db.load_faces_map()
    pprint(vectors)
    return render_template('faces.html', persons=[{'label': 1}, {'label': 2}])


@app.route('/person/<int:id>', methods=['GET'])
def person_action(id):
    # if request.method == 'POST':
    return 'Hello World {}'.format(id)


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0')
