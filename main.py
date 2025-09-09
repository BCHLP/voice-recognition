from dotenv import load_dotenv
import os
import sys
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from VoiceFingerprinter import VoiceFingerprinter
import jwt
from functools import wraps
import base64

# Create a Flask application instance
app = Flask(__name__)
app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_for=1,      # Number of proxies setting X-Forwarded-For
    x_proto=1,    # Number of proxies setting X-Forwarded-Proto
    x_host=1,     # Number of proxies setting X-Forwarded-Host
    x_prefix=1    # Number of proxies setting X-Forwarded-Prefix
)

fingerprinter=None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Get token from header
        auth_header = request.headers.get('Authorization')

        if not auth_header:
            return jsonify({'error': 'Authorization header is missing'}), 401

        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization header must start with Bearer'}), 401

        try:
            token = auth_header.split(' ')[1]
        except IndexError:
            return jsonify({'error': 'Bearer token is missing'}), 401

        # Verify the token
        passed = verify_token(token)
        if passed is False:
            return jsonify({'error': 'Token is invalid or expired'}), 401

        return f(*args, **kwargs)

    return decorated

def verify_token(token):
    try:
        payload = jwt.decode(token, os.getenv('JWT_TOKEN'), algorithms=['HS256'])
        return True
    except jwt.ExpiredSignatureError:
         return False
    except jwt.InvalidTokenError:
        return False

@app.route("/voice/register", methods=['POST'])
def generate():
    if request.method == 'POST':

        # Check if data is JSON
        audio = ''
        if request.is_json:
            data = request.get_json()
            if not data or 'audio' not in data:
                return {"error": "No audio specified"}

            audio = data['audio']

        # Check if data is form data
        elif 'audio' in request.form:
            audio = request.form['audio']
        else:
            return {"error": "Please POST a base64 encoded audio"}

        return {"fingerprint": fingerprinter.generate_fingerprint(audio)}

@app.route("/voice/compare", methods=['POST'])
def compare():
    if request.method == 'POST':

        # Check if data is JSON
        audio = ''
        embeddings = None
        if request.is_json:
            data = request.get_json()
            if not data or 'audio' not in data:
                return {"error": "No audio specified"}

            if 'embeddings' not in data:
                return {"error": "No embeddings specified"}

            audio = data['audio']
            embeddings = data['embeddings']

        # Check if data is form data
        elif 'audio' in request.form and 'embeddings' in request.form:
            audio = request.form['audio']
            embeddings = request.form['embeddings']
        else:
            return {"error": "Please POST a base64 encoded audio and embeddings"}

        return {"matches": fingerprinter.compare_audio(embeddings, audio)}

    # sarah1 = fingerprinter.generate_fingerprint("/Users/davidbelle/Projects/uni/my-voice-confirms/sarah1.wav")
    # dave4 = fingerprinter.generate_fingerprint("/Users/davidbelle/Projects/uni/my-voice-confirms/dave4.wav")
    #
    # matches = [
    #     fingerprinter.compare_audio(sarah1, "/Users/davidbelle/Projects/uni/my-voice-confirms/sarah2.wav"),
    #     fingerprinter.compare_audio(dave4, "/Users/davidbelle/Projects/uni/my-voice-confirms/dave5.wav"),
    #     fingerprinter.compare_audio(dave4, "/Users/davidbelle/Projects/uni/my-voice-confirms/sarah2.wav"),
    #     fingerprinter.compare_audio(sarah1, "/Users/davidbelle/Projects/uni/my-voice-confirms/dave5.wav")]
    #
    # print(matches)  # Expected: [True, True, False, False]


def get_cmd_args():
    port = 8080
    prevarg = ''

    for arg in sys.argv:
        if arg == '--port':
            prevarg = arg

        elif prevarg == '--port':
            port = arg
            prevarg = ''

    return {"port": port}


if __name__ == '__main__':
    load_dotenv()  # Load environment variables from .env file
    args = get_cmd_args()
    app.run(host='0.0.0.0', port=args['port'])



    # file_path='/Users/davidbelle/Projects/uni/my-voice-confirms/sarah1.wav'
    # fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
    # with open(file_path, 'rb') as wav_file:
    #     wav_bytes = wav_file.read()
    #     base64_string = base64.b64encode(wav_bytes).decode('utf-8')
    #
    # sarahs_prints = fingerprinter.generate_fingerprint(base64_string)
    #
    # file_path = '/Users/davidbelle/Projects/uni/my-voice-confirms/dave4.wav'
    # with open(file_path, 'rb') as wav_file:
    #     wav_bytes = wav_file.read()
    #     base64_string = base64.b64encode(wav_bytes).decode('utf-8')
    #
    # daves_prints = fingerprinter.generate_fingerprint(base64_string)


