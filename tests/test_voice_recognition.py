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

    def test_andrewrule_authenticates(self):
        """Test addition with positive numbers."""
        wav1 = self._read_wav_as_bytes_simple('./tests/resources/andrew-rule-1.wav')
        wav2 = self._read_wav_as_bytes_simple('./tests/resources/andrew-rule-2.wav')
        assert wav1 is not None
        assert wav2 is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(wav1)
        authenticated = fingerprinter.compare_audio(wav2, fingerprint)

        assert authenticated is True

    def test_ballen_authenticates(self):
        """Test addition with positive numbers."""
        wav1 = self._read_wav_as_bytes_simple('./tests/resources/ballen-1.wav')
        wav2 = self._read_wav_as_bytes_simple('./tests/resources/ballen-2.wav')
        assert wav1 is not None
        assert wav2 is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(wav1)
        authenticated = fingerprinter.compare_audio(wav2, fingerprint)

        assert authenticated is True

    def test_casefile_authenticates(self):
        """Test addition with positive numbers."""
        wav1 = self._read_wav_as_bytes_simple('./tests/resources/casefile-1.wav')
        wav2 = self._read_wav_as_bytes_simple('./tests/resources/casefile-2.wav')
        assert wav1 is not None
        assert wav2 is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(wav1)
        authenticated = fingerprinter.compare_audio(wav2, fingerprint)

        assert authenticated is True

    def test_hellenthomas_authenticates(self):
        """Test addition with positive numbers."""
        wav1 = self._read_wav_as_bytes_simple('./tests/resources/hellenthomas-1.wav')
        wav2 = self._read_wav_as_bytes_simple('./tests/resources/hellenthomas-2.wav')
        assert wav1 is not None
        assert wav2 is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(wav1)
        authenticated = fingerprinter.compare_audio(wav2, fingerprint)

        assert authenticated is True

    def test_ijustine_authenticates(self):
        """Test addition with positive numbers."""
        wav1 = self._read_wav_as_bytes_simple('./tests/resources/ijustine-1.wav')
        wav2 = self._read_wav_as_bytes_simple('./tests/resources/ijustine-2.wav')
        assert wav1 is not None
        assert wav2 is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(wav1)
        authenticated = fingerprinter.compare_audio(wav2, fingerprint)

        assert authenticated is True

    def test_laracasts_authenticates(self):
        """Test addition with positive numbers."""
        wav1 = self._read_wav_as_bytes_simple('./tests/resources/laracasts-1.wav')
        wav2 = self._read_wav_as_bytes_simple('./tests/resources/laracasts-2.wav')
        assert wav1 is not None
        assert wav2 is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(wav1)
        authenticated = fingerprinter.compare_audio(wav2, fingerprint)

        assert authenticated is True

    def test_lauriewired_authenticates(self):
        """Test addition with positive numbers."""
        wav1 = self._read_wav_as_bytes_simple('./tests/resources/lauriewired-1.wav')
        wav2 = self._read_wav_as_bytes_simple('./tests/resources/lauriewired-2.wav')
        assert wav1 is not None
        assert wav2 is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(wav1)
        authenticated = fingerprinter.compare_audio(wav2, fingerprint)

        assert authenticated is True

    def test_physicsgirl_authenticates(self):
        """Test addition with positive numbers."""
        wav1 = self._read_wav_as_bytes_simple('./tests/resources/physicsgirl-1.wav')
        wav2 = self._read_wav_as_bytes_simple('./tests/resources/physicsgirl-2.wav')
        assert wav1 is not None
        assert wav2 is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(wav1)
        authenticated = fingerprinter.compare_audio(wav2, fingerprint)

        assert authenticated is True

    def test_redhanded_authenticates(self):
        """Test addition with positive numbers."""
        wav1 = self._read_wav_as_bytes_simple('./tests/resources/redhanded-1.wav')
        wav2 = self._read_wav_as_bytes_simple('./tests/resources/redhanded-2.wav')
        assert wav1 is not None
        assert wav2 is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(wav1)
        authenticated = fingerprinter.compare_audio(wav2, fingerprint)

        assert authenticated is True

    def test_truecrime_authenticates(self):
        """Test addition with positive numbers."""
        wav1 = self._read_wav_as_bytes_simple('./tests/resources/truecrime-1.wav')
        wav2 = self._read_wav_as_bytes_simple('./tests/resources/truecrime-2.wav')
        assert wav1 is not None
        assert wav2 is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(wav1)
        authenticated = fingerprinter.compare_audio(wav2, fingerprint)

        assert authenticated is True


    def test_others_fail_andrewrule_authentication(self):
        """Test addition with positive numbers."""
        andrew = self._read_wav_as_bytes_simple('./tests/resources/andrew-rule-1.wav')
        ballen = self._read_wav_as_bytes_simple('./tests/resources/ballen-1.wav')
        casefile = self._read_wav_as_bytes_simple('./tests/resources/casefile-1.wav')
        hellenthomas = self._read_wav_as_bytes_simple('./tests/resources/hellen-thomas-1.wav')
        ijustine = self._read_wav_as_bytes_simple('./tests/resources/ijustine-1.wav')
        laracasts = self._read_wav_as_bytes_simple('./tests/resources/laracasts-1.wav')
        lauriewired = self._read_wav_as_bytes_simple('./tests/resources/lauriewired-1.wav')
        physicsgirl = self._read_wav_as_bytes_simple('./tests/resources/physicsgirl-1.wav')
        redhanded = self._read_wav_as_bytes_simple('./tests/resources/redhanded-1.wav')
        truecrime = self._read_wav_as_bytes_simple('./tests/resources/truecrime-1.wav')

        assert andrew is not None
        assert ballen is not None
        assert casefile is not None
        assert hellenthomas is not None
        assert ijustine is not None
        assert laracasts is not None
        assert lauriewired is not None
        assert physicsgirl is not None
        assert redhanded is not None
        assert truecrime is not None

        load_dotenv()  # Load environment variables from .env file
        fingerprinter = VoiceFingerprinter(os.getenv("HUGGING_FACE_API_KEY"))
        fingerprint = fingerprinter.generate_fingerprint(andrew)
        assert fingerprinter.compare_audio(ballen, fingerprint) is False
        assert fingerprinter.compare_audio(casefile, fingerprint) is False
        assert fingerprinter.compare_audio(hellenthomas, fingerprint) is False
        assert fingerprinter.compare_audio(ijustine, fingerprint) is False
        assert fingerprinter.compare_audio(laracasts, fingerprint) is False
        assert fingerprinter.compare_audio(lauriewired, fingerprint) is False
        assert fingerprinter.compare_audio(physicsgirl, fingerprint) is False
        assert fingerprinter.compare_audio(redhanded, fingerprint) is False
        assert fingerprinter.compare_audio(truecrime, fingerprint) is False
