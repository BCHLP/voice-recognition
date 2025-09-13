# https://www.gladia.io/blog/build-a-speaker-identification-system-for-online-meetings
# following the requirements at https://huggingface.co/pyannote/speaker-diarization-3.1

import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from pyannote.audio import Pipeline
from scipy.spatial.distance import cdist
import io
import numpy as np

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

    def generate_fingerprint(self, wav_buffer):

        waveform, sample_rate = torchaudio.load(io.BytesIO(wav_buffer))

        waveform = waveform.to(self.device)
        embedding = self.classifier.encode_batch(waveform)
        return embedding.squeeze(1).cpu().numpy()

    def compare_audio(self, wav_buffer, speaker_embedding):

        wav_bytes = io.BytesIO(wav_buffer)

        segments = self.diarization(wav_bytes)
        print("segments", segments)

        # Reset the file pointer to the beginning before loading again
        # wav_buffer.seek(0)

        # Set a threshold for similarity scores to determine when a match is considered successful
        threshold = 0.8

        waveform, sample_rate = torchaudio.load(wav_bytes)

        # Initialize variables to find the recognized speaker
        min_distance = float('inf')

        # Iterate through each segment identified in the diarization process
        for segment, label, confidence in segments.itertracks(yield_label=True):
            start_time, end_time = segment.start, segment.end

            # # Load the specific audio segment from the meeting recording
            # waveform, sample_rate = torchaudio.load(wav_bytes, num_frames=int((end_time-start_time)*sample_rate),
            #                                         frame_offset=int(start_time*sample_rate))

            # Calculate sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            #
            # # Slice the already loaded waveform (much faster than loading from disk)
            segment_waveform = waveform[:, start_sample:end_sample]
            torchaudio.save('/Users/davidbelle/Projects/uni/VoiceRecognition/myfile.wav',
                            segment_waveform,
                            sample_rate)



            # Skip segments that are too short (less than 0.5 seconds)
            if segment_waveform.shape[1] < sample_rate * 0.5:
                print(f"Skipping segment from {start_time}s to {end_time}s - too short")
                continue

            waveform = waveform.to(self.device)

            # Extract the speaker embedding from the audio segment
            embedding = self.classifier.encode_batch(waveform).squeeze(1).cpu().numpy()


            recognized_speaker_id = None

            # Compare the segment's embedding to each known speaker's embedding using cosine distance

            distances = cdist(embedding.reshape(1, -1), speaker_embedding.reshape(1, -1), metric="cosine")
            min_distance_candidate = distances.min()
            if min_distance_candidate < min_distance:
                print("min_distance_candidate", min_distance_candidate)
                print("min_distance", min_distance)
                min_distance = min_distance_candidate

        print("min_distance", min_distance)
        # Output the identified speaker and the time range they were speaking, if a match is found
        if min_distance < threshold:
            print("min_distance < threshold")
            return True

        return False
