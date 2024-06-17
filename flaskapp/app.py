import time
from datetime import datetime

from flask import Flask, request, make_response, send_file
from flask_restful import Resource, Api
from flask_cors import CORS

from CustomThread import CustomThread
from main import *
from waitress import serve

app = Flask(__name__)
CORS(app)
api = Api(app)


def generateSubtitle(inputVideo):
    milliseconds = int(time.time() * 1000)
    folder_name = str(milliseconds)
    outputAudioPath = folder_name + "/output_audio.wav"

    createAudiosDir(folder_name)
    video_to_audio(inputVideo, outputAudioPath)

    t1 = CustomThread(target=slice_wav, args=(folder_name,))
    t1.start()
    slice_video_result = t1.join()  # Wait for t1 to complete before starting t2
    t2 = CustomThread(target=asr, args=("", folder_name,slice_video_result,))
    t2.start()

    result = str(t2.join())
    removeAudiosDir(folder_name)
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


class VideosDownload(Resource):
    def post(self):
        input_text = request.get_data(as_text=True)
        videoUrl = request.args.get('videoUrl')
        add_subtitles_to_movie(videoUrl=videoUrl, subtitle_str=input_text)
        # try:
        # Check if the file exists
        file_path = "output.mp4"
        if not os.path.isfile(file_path):
            return {"message": "File not found"}, 404
        # Send the file to the client
        return send_file(file_path, as_attachment=True)
        # except Exception as e:
        #     return {"message": f"An error occurred: {str(e)}"}, 500


api.add_resource(Videos, '/api')
api.add_resource(VideosDownload, '/download')

if __name__ == '__main__':
    # app.run(debug=True, threaded=True)
    serve(app, host='0.0.0.0', port=8080)
