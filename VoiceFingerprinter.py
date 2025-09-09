# https://www.gladia.io/blog/build-a-speaker-identification-system-for-online-meetings
# following the requirements at https://huggingface.co/pyannote/speaker-diarization-3.1

import torch
import torchaudio
import base64
import io
from speechbrain.inference.speaker import EncoderClassifier
from pyannote.audio import Pipeline
from scipy.spatial.distance import cdist

class VoiceFingerprinter:
    def __init__(self, hugging_face_token):

        # Check if CUDA is available and set the device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained model for speaker embedding extraction and move it to the device
        # Note: You need to obtain an API key from Hugging Face to use this model.
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": self.device})
        self.classifier = self.classifier.to(self.device)

        # Pre-trained model for speaker diarization
        # Note: The speaker diarization model also requires an API key from Hugging Face.
        self.diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hugging_face_token)

    def __convert_to_audio_bytes(self, base64_string):

        # Decode the base64 string to bytes
        wav_bytes = base64.b64decode(base64_string)

        # Create a BytesIO object (in-memory file-like object)
        return io.BytesIO(wav_bytes)

    def generate_fingerprint(self, base64_speaker_file):

        wav_buffer = self.__convert_to_audio_bytes(base64_speaker_file)

        waveform, sample_rate = torchaudio.load(wav_buffer)
        waveform = waveform.to(self.device)
        embedding = self.classifier.encode_batch(waveform)
        return embedding.squeeze(1).cpu().numpy()

    def compare_audio(self, speaker_embedding, base64_test_audio):

        wav_buffer = self.__convert_to_audio_bytes(base64_test_audio)
        segments = self.diarization(wav_buffer)

        # Set a threshold for similarity scores to determine when a match is considered successful
        threshold = 0.8

        waveform, sample_rate = torchaudio.load(wav_buffer)

        # Iterate through each segment identified in the diarization process
        for segment, label, confidence in segments.itertracks(yield_label=True):
            start_time, end_time = segment.start, segment.end

            # Load the specific audio segment from the meeting recording
            waveform, sample_rate = torchaudio.load(wav_buffer, num_frames=int((end_time-start_time)*sample_rate),
                                                    frame_offset=int(start_time*sample_rate))

            # Skip segments that are too short (less than 0.5 seconds)
            if waveform.shape[1] < sample_rate * 0.5:
                print(f"Skipping segment from {start_time}s to {end_time}s - too short")
                continue

            waveform = waveform.to(self.device)

            # Extract the speaker embedding from the audio segment
            embedding = self.classifier.encode_batch(waveform).squeeze(1).cpu().numpy()

            # Initialize variables to find the recognized speaker
            min_distance = float('inf')
            recognized_speaker_id = None

            # Compare the segment's embedding to each known speaker's embedding using cosine distance

            distances = cdist(embedding.reshape(1, -1), speaker_embedding.reshape(1, -1), metric="cosine")
            min_distance_candidate = distances.min()
            if min_distance_candidate < min_distance:
                min_distance = min_distance_candidate

            # Output the identified speaker and the time range they were speaking, if a match is found
            if min_distance < threshold:
                return True

        return False
