import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import io
import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats
import warnings
import whisper
import librosa

class VoiceFingerprinter:
    def __init__(self, hugging_face_token=None, threshold=0.42, min_segment_length=1.0):
        """
        Initialize Voice Fingerprinter

        Args:
            hugging_face_token: HuggingFace token (not needed for speechbrain model)
            threshold: Cosine distance threshold for speaker verification (lower = stricter)
            min_segment_length: Minimum segment length in seconds
        """
        # Check if CUDA is available and set the device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load pre-trained model for speaker embedding extraction
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

        self.threshold = threshold
        self.min_segment_length = min_segment_length
        self.sample_rate = 16000  # Standard sample rate for the model

    def match_audio_with_text(self, wav_buffer, text):

        model = whisper.load_model("medium")

        audio_np, sr = librosa.load(io.BytesIO(wav_buffer), sr=16000, mono=True)

        result = model.transcribe(audio_np, language='en')
        print(result["text"])

    def _preprocess_audio(self, waveform, sample_rate):
        """
        Preprocess audio: resample and normalize
        """
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        # Normalize audio to prevent volume variations from affecting results
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        # Optional: Apply simple high-pass filter if available (skip the problematic scipy version for now)
        try:
            if hasattr(torchaudio.transforms, 'HighpassBiquad') and waveform.shape[1] > 1000:
                highpass = torchaudio.transforms.HighpassBiquad(
                    sample_rate=self.sample_rate,
                    cutoff_freq=80
                )
                waveform = highpass(waveform)
        except Exception as e:
            # Skip filtering if it causes issues
            pass

        return waveform.to(self.device)

    def generate_fingerprint(self, wav_buffer):
        """
        Generate speaker embedding from audio buffer
        """
        try:
            waveform, sample_rate = torchaudio.load(io.BytesIO(wav_buffer))

            # Preprocess audio
            waveform = self._preprocess_audio(waveform, sample_rate)

            # Generate embedding
            with torch.no_grad():
                embedding = self.classifier.encode_batch(waveform)

            return embedding.squeeze().cpu().numpy()

        except Exception as e:
            print(f"Error generating fingerprint: {e}")
            return None

    def _split_audio_into_segments(self, waveform, segment_duration=2.0, overlap=0.5):
        """
        Split long audio into overlapping segments for better analysis
        """
        segment_samples = int(segment_duration * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        step_size = segment_samples - overlap_samples

        segments = []
        total_samples = waveform.shape[1]

        for start in range(0, total_samples - segment_samples + 1, step_size):
            end = start + segment_samples
            if end > total_samples:
                end = total_samples
                start = max(0, end - segment_samples)

            segment = waveform[:, start:end]
            if segment.shape[1] >= self.sample_rate * self.min_segment_length:
                segments.append(segment)

        # If audio is shorter than segment_duration, use the whole thing
        if not segments and total_samples >= self.sample_rate * self.min_segment_length:
            segments.append(waveform)

        return segments

    def compare_audio(self, wav_buffer, reference_embedding, use_statistical_analysis=True):
        """
        Compare audio against reference embedding with improved statistical analysis

        Args:
            wav_buffer: Audio data as bytes
            reference_embedding: Reference speaker embedding
            use_statistical_analysis: Whether to use multiple segments for statistical analysis

        Returns:
            dict: Contains 'is_match', 'confidence', 'min_distance', 'mean_distance', 'distances'
        """
        try:
            wav_bytes = io.BytesIO(wav_buffer)
            waveform, sample_rate = torchaudio.load(wav_bytes)

            # Preprocess audio
            waveform = self._preprocess_audio(waveform, sample_rate)

            # Check if audio is long enough
            if waveform.shape[1] < self.sample_rate * self.min_segment_length:
                print(f"Audio too short: {waveform.shape[1] / self.sample_rate:.2f}s")
                return {
                    'is_match': False,  # Already Python boolean
                    'confidence': 0.0,
                    'min_distance': float('inf'),
                    'mean_distance': float('inf'),
                    'distances': [],
                    'reason': 'Audio too short'
                }

            if use_statistical_analysis:
                # Split into segments for statistical analysis
                segments = self._split_audio_into_segments(waveform)

                if not segments:
                    return {
                        'is_match': False,  # Already Python boolean
                        'confidence': 0.0,
                        'min_distance': float('inf'),
                        'mean_distance': float('inf'),
                        'distances': [],
                        'reason': 'No valid segments found'
                    }

                distances = []

                # Process each segment
                for i, segment in enumerate(segments):
                    try:
                        with torch.no_grad():
                            embedding = self.classifier.encode_batch(segment)
                        embedding = embedding.squeeze().cpu().numpy()

                        # Calculate cosine distance
                        distance = cdist(
                            embedding.reshape(1, -1),
                            reference_embedding.reshape(1, -1),
                            metric="cosine"
                        )[0, 0]

                        distances.append(distance)
                        print(f"Segment {i + 1}: distance = {distance:.4f}")

                    except Exception as e:
                        print(f"Error processing segment {i + 1}: {e}")
                        continue

                if not distances:
                    return {
                        'is_match': False,  # Already Python boolean
                        'confidence': 0.0,
                        'min_distance': float('inf'),
                        'mean_distance': float('inf'),
                        'distances': [],
                        'reason': 'No segments processed successfully'
                    }

                # Statistical analysis
                distances = np.array(distances)
                min_distance = np.min(distances)
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)

                print(f"Statistics: min={min_distance:.4f}, mean={mean_distance:.4f}, std={std_distance:.4f}")

                # Decision logic: More flexible approach with multiple criteria
                # Primary criterion: minimum distance should be reasonable
                primary_match = min_distance < self.threshold

                # Secondary: allow higher distances if consistency is very good
                # Good consistency (low std) indicates same speaker across segments
                excellent_consistency = std_distance < 0.06  # Very consistent
                good_consistency = std_distance < 0.10  # Reasonably consistent

                # Flexible thresholds based on consistency
                if excellent_consistency:
                    secondary_threshold = self.threshold + 0.15  # Allow up to 0.57
                elif good_consistency:
                    secondary_threshold = self.threshold + 0.10  # Allow up to 0.52
                else:
                    secondary_threshold = self.threshold + 0.05  # Allow up to 0.47

                secondary_match = (min_distance < secondary_threshold and
                                   mean_distance < secondary_threshold + 0.10)

                # Accept if either primary matches OR secondary criteria with good consistency
                is_match = primary_match or (secondary_match and good_consistency)

                # Calculate confidence score based on multiple factors
                distance_confidence = max(0, 1 - (min_distance / (self.threshold + 0.3)))
                consistency_confidence = max(0, 1 - (std_distance / 0.15))
                confidence = (distance_confidence + consistency_confidence) / 2

            else:
                # Simple single-embedding comparison
                with torch.no_grad():
                    embedding = self.classifier.encode_batch(waveform)
                embedding = embedding.squeeze().cpu().numpy()

                min_distance = cdist(
                    embedding.reshape(1, -1),
                    reference_embedding.reshape(1, -1),
                    metric="cosine"
                )[0, 0]

                mean_distance = min_distance
                distances = [min_distance]
                is_match = min_distance < self.threshold
                confidence = max(0, 1 - (min_distance / (self.threshold + 0.2)))

            print(f"Final decision: {'MATCH' if is_match else 'NO MATCH'} "
                  f"(confidence: {confidence:.2f})")

            return {
                'is_match': bool(is_match),  # Convert numpy boolean to Python boolean
                'confidence': float(confidence),
                'min_distance': float(min_distance),
                'mean_distance': float(mean_distance),
                'distances': [float(d) for d in distances],
                'reason': 'Analysis complete'
            }

        except Exception as e:
            print(f"Error in compare_audio: {e}")
            return {
                'is_match': False,  # Already Python boolean
                'confidence': 0.0,
                'min_distance': float('inf'),
                'mean_distance': float('inf'),
                'distances': [],
                'reason': f'Error: {str(e)}'
            }

    def calibrate_threshold(self, positive_samples, negative_samples):
        """
        Calibrate threshold based on positive and negative samples

        Args:
            positive_samples: List of audio buffers from the target speaker
            negative_samples: List of audio buffers from different speakers

        Returns:
            dict: Recommended threshold and performance metrics
        """
        print("Calibrating threshold...")

        # Generate reference embedding from positive samples
        reference_embeddings = []
        for sample in positive_samples:
            embedding = self.generate_fingerprint(sample)
            if embedding is not None:
                reference_embeddings.append(embedding)

        if not reference_embeddings:
            raise ValueError("No valid positive samples for calibration")

        # Use mean of positive samples as reference
        reference_embedding = np.mean(reference_embeddings, axis=0)

        # Test different thresholds
        thresholds = np.arange(0.1, 0.8, 0.02)
        best_threshold = 0.25
        best_f1 = 0

        results = []

        for threshold in thresholds:
            self.threshold = threshold

            # Test on positive samples
            positive_results = []
            for sample in positive_samples:
                result = self.compare_audio(sample, reference_embedding, use_statistical_analysis=False)
                positive_results.append(result['is_match'])

            # Test on negative samples
            negative_results = []
            for sample in negative_samples:
                result = self.compare_audio(sample, reference_embedding, use_statistical_analysis=False)
                negative_results.append(not result['is_match'])  # True if correctly rejected

            # Calculate metrics
            true_positive_rate = np.mean(positive_results)
            true_negative_rate = np.mean(negative_results)

            # F1 score balances precision and recall
            precision = true_positive_rate
            recall = true_positive_rate
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            results.append({
                'threshold': threshold,
                'true_positive_rate': true_positive_rate,
                'true_negative_rate': true_negative_rate,
                'f1_score': f1
            })

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold = best_threshold
        print(f"Best threshold: {best_threshold:.3f} (F1 score: {best_f1:.3f})")

        return {
            'recommended_threshold': best_threshold,
            'best_f1_score': best_f1,
            'calibration_results': results
        }