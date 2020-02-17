'''
 * @Description: This is restful server
 * @Author: Meng Zeyu @ https://github.com/mengzeyu/nlp-project-01/blob/master/restful.py
 * @Date: 2019-12-04
 * @Adopted: Yang Jieying
'''
from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from flask import redirect, url_for
from flask_cors import CORS
from Automatic_entext_summary import AutomaticTextSummarizer
from Automatic_entext_summary import load_model

class HelloWorld(Resource):
    def post(self):
        args=parser.parse_args()
        title = args['title']
        content = args['content']
        print(title,content)
        result = automatic_text_summarizer.summarize(title, content)
        print(result['summarization'])
        return jsonify(summarization= result['summarization'])

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('title', type=str,location='form')
parser.add_argument('content', type=str, location='form')

path_en = '../model/word2vec_enfull.model'

model = load_model(path_en)
	
automatic_text_summarizer = AutomaticTextSummarizer(model)
automatic_text_summarizer.set_n_neighbors(5)
automatic_text_summarizer.set_summarize_sentences_size(0.4)

api.add_resource(HelloWorld, '/testing')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444 , debug=True)

