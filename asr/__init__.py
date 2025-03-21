import gevent.monkey
gevent.monkey.patch_all()
from email.mime.multipart import MIMEMultipart
from email.message import Message
from .model_map import get_model_for_lang
import json
import os
import struct
import requests
import io
import wave
import time
import audioop
import logging
from speex import SpeexDecoder
from flask import Flask, request, Response, abort
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('rebble-asr')

decoder = SpeexDecoder(1)
app = Flask(__name__)

model_size = "guillaumekln/faster-whisper-small"
logger.info(f"Initializing WhisperModel with size: {model_size}")
model = WhisperModel(model_size, device="cpu", compute_type="int8")
logger.info("Model initialization complete")

#AUTH_URL = "https://auth.rebble.io"


# We know gunicorn does this, but it doesn't *say* it does this, so we must signal it manually.
@app.before_request
def handle_chunking():
    request.environ['wsgi.input_terminated'] = 1


def parse_chunks(stream):
    boundary = b'--' + request.headers['content-type'].split(';')[1].split('=')[1].encode('utf-8').strip()  # super lazy/brittle parsing.
    this_frame = b''
    while True:
        content = stream.read(4096)
        this_frame += content
        end = this_frame.find(boundary)
        if end > -1:
            frame = this_frame[:end]
            this_frame = this_frame[end + len(boundary):]
            if frame != b'':
                try:
                    header, content = frame.split(b'\r\n\r\n', 1)
                except ValueError:
                    continue
                yield content[:-2]
        if content == b'':
            print("End of input.")
            break


@app.route('/heartbeat')
def heartbeat():
    return 'asr'

@app.route('/NmspServlet/', methods=["POST"])
def recognise():
    request_id = f"req-{int(time.time())}-{os.urandom(2).hex()}"
    logger.info(f"[{request_id}] Received recognition request")
    
    stream = request.stream
    
    chunks = list(parse_chunks(stream))
    logger.info(f"[{request_id}] Parsed {len(chunks)} chunks, using chunks[3:]")
    chunks = chunks[3:]
    
    # Instead of AudioSegment.empty(), use a bytearray to accumulate PCM data
    pcm_data = bytearray()

    if len(chunks) > 15:
        logger.info(f"[{request_id}] Trimming chunks from {len(chunks)} to chunks[12:-3]")
        chunks = chunks[12:-3]
    
    logger.info(f"[{request_id}] Processing {len(chunks)} audio chunks")
    for i, chunk in enumerate(chunks):
        decoded = decoder.decode(chunk)
        # Boosting the audio volume
        decoded = audioop.mul(decoded, 2, 7)
        # Directly append decoded audio bytes
        pcm_data.extend(decoded)
        if i % 5 == 0:  # Log every 5 chunks to avoid excessive logging
            logger.debug(f"[{request_id}] Processed chunk {i+1}/{len(chunks)}")
    
    logger.info(f"[{request_id}] Audio decoding complete, total PCM size: {len(pcm_data)} bytes")

    # Create WAV file in memory
    logger.info(f"[{request_id}] Creating WAV file in memory")
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm_data)
    
    wav_size = wav_buffer.tell()
    wav_buffer.seek(0)
    logger.info(f"[{request_id}] WAV file created, size: {wav_size} bytes")
    
    # Save WAV to a temporary file for transcription
    import tempfile
    
    logger.info(f"[{request_id}] Saving WAV to temporary file")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav.write(wav_buffer.getvalue())
        temp_wav_path = temp_wav.name
    
    # Continue with the transcription
    logger.info(f"[{request_id}] Starting transcription with Whisper model")
    transcription_start = time.time()
    
    try:
        # Pass WAV file path directly to the model
        logger.info(f"[{request_id}] Using WAV file for transcription: {temp_wav_path}")
        segments, info = model.transcribe(temp_wav_path, beam_size=5)
        logger.info(f"[{request_id}] Detected language '{info.language}' with probability {info.language_probability:.4f}")
        segments = list(segments)
        transcription_time = time.time() - transcription_start
        logger.info(f"[{request_id}] Transcription completed in {transcription_time:.2f}s")

    except Exception as e:
        logger.error(f"[{request_id}] Transcription error: {e}")
        # Add more detailed error logging
        import traceback
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        return Response("Error during transcription", status=500)
    finally:
        # Clean up temporary file
        logger.debug(f"[{request_id}] Cleaning up temporary file")
        try:
            os.unlink(temp_wav_path)
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to clean up temp file: {e}")
    
    transcript = segments[0].text if len(segments) > 0 else ""
    logger.info(f"[{request_id}] Transcript: '{transcript}'")
    
    words = []
    for word in transcript.split():
        words.append({
            'word': word,
            'confidence': 1.0
        })

    # Now create a MIME multipart response
    logger.info(f"[{request_id}] Forming response with {len(words)} words")
    parts = MIMEMultipart()
    response_part = Message()
    response_part.add_header('Content-Type', 'application/JSON; charset=utf-8')

    if len(words) > 0:
        response_part.add_header('Content-Disposition', 'form-data; name="QueryResult"')
        # Append the no-space marker and uppercase the first character
        words[0]['word'] += '\\*no-space-before'
        words[0]['word'] = words[0]['word'][0].upper() + words[0]['word'][1:]
        payload = json.dumps({'words': [words]})
        logger.info(f"[{request_id}] Created QueryResult payload")
    else:
        response_part.add_header('Content-Disposition', 'form-data; name="QueryRetry"')
        payload = json.dumps({
            "Cause": 1,
            "Name": "AUDIO_INFO",
            "Prompt": "Sorry, speech not recognized. Please try again."
        })
        logger.info(f"[{request_id}] Created QueryRetry payload - no words recognized")

    response_part.set_payload(payload)
    parts.attach(response_part)

    parts.set_boundary('--Nuance_NMSP_vutc5w1XobDdefsYG3wq')
    response_text = '\r\n' + parts.as_string().split("\n", 3)[3].replace('\n', '\r\n')
    response = Response(response_text)
    response.headers['Content-Type'] = f'multipart/form-data; boundary={parts.get_boundary()}'
    logger.info(f"[{request_id}] Sending response, size: {len(response_text)} bytes")
    return response

