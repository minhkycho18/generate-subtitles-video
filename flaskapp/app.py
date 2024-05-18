from flask import Flask, request, send_file, make_response, jsonify
from flask_restful import Resource, Api, marshal_with, fields
from flask_sqlalchemy import SQLAlchemy
from main import *

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todo.db'
db = SQLAlchemy(app)

api = Api(app)

taskFields = {
    'id': fields.Integer,
    'name': fields.String,
}

fakeDatabase = {
    1: {'name': 'Clean car'},
    2: {'name': 'Write blog'},
    3: {'name': 'Start stream'},
}

def generateSubtitle(inputVideo):
    result = ""
    video_to_audio(inputVideo, OUTPUT_AUDIO)
    createAudiosDir()
    t1 = threading.Thread(target=slice_wav)
    t2 = threading.Thread(target=asr)
    t1.start()
    t1.join()  # Wait for t1 to complete before starting t2
    t2.start()

    t2.join()
    removeAudiosDir()
    with open('myfile.vtt', 'r') as file:
        result = file.read()
    return result

class Items(Resource):
    # @marshal_with(taskFields)
    def get(self):
        response = make_response(generateSubtitle("videoUrl"), 200)
        response.mimetype = "text/plain"
        return response

    # @marshal_with(taskFields)
    def post(self):
        json_data = request.get_json(force=True)
        videoUrl = json_data['url']
        response = make_response(generateSubtitle(videoUrl), 200)
        response.mimetype = "text/plain"
        return response
        # return jsonify(url=un)


class Item(Resource):
    @marshal_with(taskFields)
    def get(self, pk):
        return generateSubtitle()

    @marshal_with(taskFields)
    def put(self, pk):
        data = request.json
        task = Task.query.filter_by(id=pk).first()
        task.name = data['name']
        db.session.commit()
        return task

    @marshal_with(taskFields)
    def delete(self, pk):
        task = Task.query.filter_by(id=pk).first()
        db.session.delete(task)
        db.session.commit()
        tasks = Task.query.all()
        return tasks


class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return self.name


api.add_resource(Items, '/')

api.add_resource(Item, '/<int:pk>')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
