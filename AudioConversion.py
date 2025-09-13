import base64
import io
from pydub import AudioSegment

class AudioConversion:
    def convert_webm_to_wav(self, webm_bytes):
        # Create BytesIO objects
        webm_io = io.BytesIO(webm_bytes)
        wav_io = io.BytesIO()

        # Load webm and export to wav in memory
        audio = AudioSegment.from_file(webm_io, format="webm")
        audio.export(wav_io, format="wav")

        # Get the wav bytes
        wav_bytes = wav_io.getvalue()

        # Clean up
        webm_io.close()
        wav_io.close()

        return wav_bytes

    def base64_to_audio_bytes(self, base64_string):

        # Decode the base64 string to bytes
        wav_bytes = base64.b64decode(base64_string)

        # Create a BytesIO object (in-memory file-like object)
        return wav_bytes
