from flask import Flask, request, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
from CustomThread import CustomThread
from main import *
from waitress import serve

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
api = Api(app)


def generateSubtitle(inputVideo):
    createAudiosDir()
    video_to_audio(inputVideo, OUTPUT_AUDIO_PATH)
    t1 = CustomThread(target=slice_wav)
    t2 = CustomThread(target=asr, args=("",))
    t1.start()
    t1.join()  # Wait for t1 to complete before starting t2
    t2.start()

    result = str(t2.join())
    removeAudiosDir()
    return result


class Videos(Resource):
    def get(self):
        response = make_response(generateSubtitle("videoUrl"), 200)
        response.mimetype = "text/plain"
        return response

    def post(self):
        json_data = request.get_json(force=True)
        videoUrl = json_data['url']
        response = make_response(generateSubtitle(videoUrl), 200)
        response.mimetype = "text/plain"
        return response

api.add_resource(Videos, '/api/')

if __name__ == '__main__':
    # app.run(debug=True, threaded=True)
    serve(app, host='0.0.0.0', port=8080)
