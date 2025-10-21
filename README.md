# Voice Authentication API

A Flask-based voice authentication API that uses speaker recognition and voice biometric fingerprinting to verify user identity. Built with SpeechBrain's ECAPA-VOXCELEB model for high-accuracy speaker verification.

## Features

- Voice fingerprint generation and enrollment
- Speaker verification and comparison
- Base64-encoded audio input support (WebM to WAV conversion)
- JWT token authentication
- Statistical analysis for improved accuracy
- CUDA/GPU acceleration support
- RESTful API interface

## System Requirements

### Hardware
- **Recommended**: CUDA-capable GPU for optimal performance
- **Minimum**: 8GB RAM (16GB+ recommended for GPU usage)
- CPU-only mode is supported but significantly slower

### Software
- Python 3.8+
- FFmpeg (required for audio conversion)
- CUDA Toolkit (optional, for GPU acceleration)

## Installation

### 1. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

### 2. Clone the Repository

```bash
git clone https://github.com/BCHLP/voice-recognition
cd voice-recognition
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies include:
- `flask` - Web framework
- `gunicorn` - Production WSGI server
- `speechbrain` - Speaker recognition models
- `torchaudio` - Audio processing with PyTorch
- `pydub` - Audio format conversion
- `pyannote.audio` - Audio analysis
- `PyJWT` - Token authentication
- `librosa`, `scipy`, `numpy` - Audio processing and analysis

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
HUGGING_FACE_API_KEY=your_huggingface_token_here
JWT_TOKEN=your_secret_jwt_key_here
```

**Note:** The Hugging Face API key may not be strictly required for the SpeechBrain model, but it's recommended for accessing additional models if needed.

### 5. Generate JWT Token

Use the provided script to generate a secure JWT token:

```bash
python generate-token.py
```
Use the token to authenticate API requests
## Usage

### Development Mode (Flask)

Run the application using Flask's development server:

```bash
python main.py
```

By default, the server runs on `http://0.0.0.0:8080`

To specify a custom port:
```bash
python main.py --port 5000
```

### Production Mode (Gunicorn)

For production deployments, use Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:8080 main:app
```

Options:
- `-w 4`: Number of worker processes (adjust based on CPU cores)
- `-b 0.0.0.0:8080`: Bind address and port
- `--timeout 120`: Increase timeout for long-running requests

## API Endpoints

### 1. Register Voice (Generate Fingerprint)

**Endpoint:** `POST /voice/register`

**Description:** Generates a voice fingerprint (embedding) from audio data.

**Request Body:**
```json
{
  "audio": "base64_encoded_audio_data"
}
```

**Response:**
```json
{
  "embeddings": "base64_encoded_fingerprint"
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/voice/register \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQRChYECGFOAZwH/////////FUmpZpkq17GDD0JATYCGQ2hyb..."
  }'
```

### 2. Compare Voice (Verify Authentication)

**Endpoint:** `POST /voice/compare`

**Description:** Compares audio against a stored voice fingerprint to verify identity.

**Request Body:**
```json
{
  "audio": "base64_encoded_audio_data",
  "embeddings": "base64_encoded_reference_fingerprint"
}
```

**Response:**
```json
{
  "authenticated": true,
  "didspeakermatch": true
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/voice/compare \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQRChYECGFOAZwH/////////FUmpZpkq17GDD0JATYCGQ2hyb...",
    "embeddings": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA..."
  }'
```

## Audio Format Support

The API accepts base64-encoded audio in the following formats:
- WebM (automatically converted to WAV)
- WAV
- Other formats supported by FFmpeg/pydub

Audio is automatically:
1. Decoded from base64
2. Converted to WAV format (16kHz, mono)
3. Normalized and preprocessed
4. Processed for speaker embedding extraction

## Authentication

Currently, the codebase includes JWT authentication decorators (`@token_required`) but they are not applied to the endpoints. To enable authentication:

1. Add the `@token_required` decorator to protected endpoints
2. Include the JWT token in request headers:

```bash
curl -X POST http://localhost:8080/voice/register \
  -H "Authorization: Bearer your_jwt_token_here" \
  -H "Content-Type: application/json" \
  -d '{"audio": "..."}'
```

## How It Works

### Voice Fingerprinting
1. Audio is preprocessed (resampled to 16kHz, normalized)
2. SpeechBrain's ECAPA-TDNN model extracts a 192-dimensional embedding
3. The embedding is a unique "fingerprint" of the speaker's voice characteristics

### Speaker Verification
1. Audio is split into overlapping segments for statistical analysis
2. Each segment generates an embedding
3. Cosine distance is calculated between test and reference embeddings
4. Statistical analysis (minimum, mean, standard deviation) determines match
5. Adaptive thresholds based on consistency improve accuracy

### Performance Tuning
- **Threshold**: Default 0.42 (lower = stricter matching)
- **Min Segment Length**: 1.0 second minimum audio duration
- **Segment Analysis**: 2-second segments with 50% overlap

## Testing

Run the test suite:

```bash
pytest
```

## Troubleshooting

### CUDA/GPU Issues
If you encounter CUDA errors, the system will automatically fall back to CPU:
```python
Using device: cpu
```

To disable NNPACK (if causing issues):
```python
torch.backends.nnpack.enabled = False  # Already set in VoiceFingerprinter
```

### Audio Too Short Error
Ensure audio samples are at least 1 second long. Shorter audio may not generate reliable fingerprints.

### FFmpeg Not Found
Ensure FFmpeg is installed and available in your system PATH:
```bash
ffmpeg -version
```

### Memory Issues
If running on limited hardware:
- Reduce the number of Gunicorn workers
- Use CPU-only mode
- Process shorter audio segments

## Project Structure

```
voice-recognition/
├── main.py                 # Flask application and API endpoints
├── VoiceFingerprinter.py   # Core voice fingerprinting logic
├── AudioConversion.py      # Audio format conversion utilities
├── generate-token.py       # JWT token generation script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
└── README.md              # This file
```

