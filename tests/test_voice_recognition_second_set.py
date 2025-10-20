import pytest
from dotenv import load_dotenv
import os
from ..VoiceFingerprinter import VoiceFingerprinter
class TestBasicOperations:

    def _read_wav_as_bytes_simple(self, filename):
        """
        Read entire WAV file as raw bytes.
        Returns the complete file content including headers.
        """
        with open(filename, 'rb') as wav_file:
            byte_array = wav_file.read()
        return byte_array

    """Test class for basic mathematical operations."""

    def test_for_positives(self):
        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))

        passes = 0
        for i in range(1, 21):

            wav1 = self._read_wav_as_bytes_simple('./tests/resources/'+str(i)+'.1.opus.wav')
            wav2 = self._read_wav_as_bytes_simple('./tests/resources/'+str(i)+'.2.opus.wav')
            assert wav1 is not None
            assert wav2 is not None

            fingerprint = fingerprinter.generate_fingerprint(wav1)
            results = fingerprinter.compare_audio(wav2, fingerprint)

            if results['is_match']:
                passes += 1

        assert passes is 20

    def test_for_negatives(self):

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))

        passes = 0
        total = 0
        for i in range(1, 21):

            sample1 = self._read_wav_as_bytes_simple('./tests/resources/'+str(i)+'.1.opus.wav')
            sample2 = self._read_wav_as_bytes_simple('./tests/resources/'+str(i)+'.2.opus.wav')

            for t in range(1, 21):
                if t == i:
                    continue

                against1 = self._read_wav_as_bytes_simple('./tests/resources/' + str(t) + '.1.opus.wav')
                against2 = self._read_wav_as_bytes_simple('./tests/resources/' + str(t) + '.2.opus.wav')

                fingerprint1 = fingerprinter.generate_fingerprint(against1)
                fingerprint2 = fingerprinter.generate_fingerprint(against2)
                results1 = fingerprinter.compare_audio(sample1, fingerprint1)
                results2 = fingerprinter.compare_audio(sample1, fingerprint2)
                results3 = fingerprinter.compare_audio(sample2, fingerprint1)
                results4 = fingerprinter.compare_audio(sample2, fingerprint2)

                if not results1['is_match']:
                    passes += 1

                if not results2['is_match']:
                    passes += 1

                if not results3['is_match']:
                    passes += 1

                if not results4['is_match']:
                    passes += 1

                total += 4

        assert passes == total
