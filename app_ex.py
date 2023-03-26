import os
import re
import tempfile
import threading
import uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from num2words import num2words
from pydub import AudioSegment
import torchaudio
from speechbrain.pretrained import HIFIGAN, Tacotron2
import whisper
from cachetools import cached, LRUCache
import fasteners
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi import FastAPI, Body, Request
import asyncio
import subprocess



class TTSInput(BaseModel):
    text: str


# FastAPI app
app = FastAPI()

# Load TTS model
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

# TTS file prefix
speech_tts_prefix = "speech-tts-"
wav_suffix = ".wav"
opus_suffix = ".opus"

# Load transcription model
model = whisper.load_model("base")

# Create lock object
lock_file = os.path.join(tempfile.gettempdir(), "tmp-lockfile")
lock = fasteners.InterProcessLock(lock_file)

# Clean temporary files (called every hour)
def clean_tmp():
    with lock:
        tmp_dir = tempfile.gettempdir()
        for file in os.listdir(tmp_dir):
            if file.startswith(speech_tts_prefix):
                os.remove(os.path.join(tmp_dir, file))
        print("[Speech REST API] Temporary files cleaned!")


# Preprocess text to replace numerals with words
def preprocess_text(text):
    text = re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
    return text

# Run TTS and save file
# Returns the path to the file
@cached(cache=LRUCache(maxsize=1000))
def run_tts_and_save_file(sentence):
    # Running the TTS
    mel_outputs, mel_length, alignment = tacotron2.encode_batch([sentence])

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_outputs)

    # Get temporary directory
    tmp_dir = tempfile.gettempdir()

    # Save wav to temporary file
    tmp_path_wav = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + wav_suffix)
    torchaudio.save(tmp_path_wav, waveforms.squeeze(1), 22050)
    return tmp_path_wav


# TTS endpoint
@app.post('/tts')
async def generate_tts(input_data: TTSInput):
    text = input_data.text
    print(text)
    text = text.replace("'", "")
    text = text.replace('"', "")

    # Preprocess text to replace numerals with words
    text = preprocess_text(text)

    # Split text by . ? !
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

    # Trim sentences
    sentences = [sentence.strip() for sentence in sentences]

    # Remove empty sentences
    sentences = [sentence for sentence in sentences if sentence]

    # Logging
    print("[Speech REST API] Got request: length (" + str(len(text)) + "), sentences (" + str(len(sentences)) + ")")

    # Run TTS for each sentence
    output_files = []

    for sentence in sentences:
        print("[Speech REST API] Generating TTS: " + sentence)
        loop = asyncio.get_event_loop()
        tmp_path_wav = await loop.run_in_executor(None, run_tts_and_save_file, sentence)
        output_files.append(tmp_path_wav)


    # Concatenate and convert files using ffmpeg
    output_files_str = ' '.join(f'-i {f}' for f in output_files)
    print(output_files_str)
    tmp_dir = tempfile.gettempdir()
    tmp_path_opus = os.path.join(tmp_dir, speech_tts_prefix + str(uuid.uuid4()) + opus_suffix)

    if len(output_files) > 1:
        filter_complex = f'[0:a][1:a]concat=n={len(output_files)}:v=0:a=1[out]'
        if len(output_files) > 2:
            filter_complex = f'[0:a][1:a][2:a]concat=n={len(output_files)}:v=0:a=1[out]'
        cmd = f'ffmpeg {output_files_str} -filter_complex "{filter_complex}" -map "[out]" -acodec libopus -b:a 64k -f opus "{tmp_path_opus}"'
    else:
        cmd = f'ffmpeg {output_files_str} -acodec libopus -b:a 64k -f opus "{tmp_path_opus}"'


    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e}")
    # handle the error, e.g., return an error response or raise an exception

    # Delete tmp files
    #for file in output_files:
    #    os.remove(file)

    # Return file response
    return FileResponse(tmp_path_opus, media_type="audio/ogg")




#Transcribe endpoint
@app.post('/transcribe')
async def transcribe(audio: UploadFile = File(...)):
    # Save audio file into tmp folder
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, str(uuid.uuid4()))
    with open(tmp_path, "wb") as f:
        f.write(await audio.read())

    try:
        # Load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(tmp_path)
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detect the spoken language
        _, probs = model.detect_language(mel)
        language = max(probs, key=probs.get)

        # Decode the audio
        result = whisper.transcribe(model, tmp_path)
        text_result = result["text"]
        text_result_trim = text_result.strip()

        # Delete tmp file
        os.remove(tmp_path)

        return {
            'language': language,
            'text': text_result_trim
        }
    except Exception as e:
        # Delete tmp file in case of error
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail="Error processing audio file: " + str(e))


#Health endpoint
@app.get('/health')
async def health():
    return {'status': 'ok'}

#Clean endpoint
@app.get('/clean')
async def clean():
    clean_tmp()
    return {'status': 'ok'}

#Start the timer to clean temporary files
clean_tmp()

#Start the timer to clean temporary files every hour
tmp_clean_thread = threading.Timer(3600.0, clean_tmp)
tmp_clean_thread.daemon = True
tmp_clean_thread.start()
print("[Speech REST API] Temporary file cleaning thread started!")

#Start the FastAPI server
print("[Speech REST API] Starting server...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_ex:app", host="0.0.0.0", port=3000, log_level="info")

