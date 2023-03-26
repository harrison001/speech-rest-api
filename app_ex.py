import os
import re
import tempfile
import asyncio
import subprocess
from fastapi import FastAPI, File, UploadFile
from num2words import num2words
import torchaudio
from speechbrain.pretrained import HIFIGAN, Tacotron2
import whisper
from aiocache import cached
from aiocache.backends.memory import SimpleMemoryCache
from fastapi.responses import FileResponse
from pydantic import BaseModel
import aiofiles
from aiofiles.os import remove as async_remove
from aiofiles.tempfile import NamedTemporaryFile
from asyncio import Semaphore
import logging
import uvicorn
import uuid


# FastAPI app
app = FastAPI()

async def on_startup():
    asyncio.create_task(clean_tmp_periodically(3600))

app.add_event_handler("startup", on_startup)

class TTSInput(BaseModel):
    text: str

async def clean_tmp():
    tmp_dir = tempfile.gettempdir()
    for file in os.listdir(tmp_dir):
        if file.startswith(speech_tts_prefix):
            await asyncio.to_thread(os.remove, os.path.join(tmp_dir, file))
    logging.info("[Speech REST API] Temporary files cleaned!")


async def clean_tmp_periodically(interval: int):
    while True:
        await clean_tmp()
        await asyncio.sleep(interval)

# Preprocess text to replace numerals with words
def preprocess_text(text):
    text = re.sub(r'\d+', lambda m: num2words(int(m.group(0))), text)
    return text

# Run TTS and save file
# Returns the path to the file
@cached(ttl=3600, cache=SimpleMemoryCache, key_from_args=lambda args, kwargs: args[0])
async def run_tts_and_save_file(sentence):
    # Running the TTS
    mel_outputs, mel_length, alignment = tacotron2.encode_batch([sentence])

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_outputs)

    # Save wav to temporary file
    async with NamedTemporaryFile(prefix=speech_tts_prefix, suffix=wav_suffix, delete=False) as tmp_file:
        tmp_path_wav = tmp_file.name
        await asyncio.to_thread(torchaudio.save, tmp_path_wav, waveforms.squeeze(1), 22050)
        return tmp_path_wav

# TTS endpoint
@app.post('/tts')
async def generate_tts(input_data: TTSInput):
    async with request_semaphore:
        text = input_data.text
        logging.info(text)
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
    logging.info("[Speech REST API] Got request: length (" + str(len(text)) + "), sentences (" + str(len(sentences)) + ")")

    # Run TTS for each sentence
    output_files = []

    for sentence in sentences:
        output_files.append(await run_tts_and_save_file(sentence))

    # Concatenate and convert files using ffmpeg
    output_files_str = ' '.join(f'-i {f}' for f in output_files)
    logging.info(output_files_str)
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
        await async_run_command(cmd)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during ffmpeg execution: {e}")

    # Return file response
    return FileResponse(tmp_path_opus, media_type="audio/ogg")

async def async_run_command(cmd):
    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    return stdout


#Transcribe endpoint
@app.post('/transcribe')
async def transcribe(audio: UploadFile = File(...)):
    async with request_semaphore:
        # Use asynchronous context manager to save audio file to a temporary folder
        async with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            async with aiofiles.open(tmp_path, "wb") as f:
                await f.write(await audio.read())

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

        # Remove the temporary files
        await async_remove(tmp_path)

        return {
            'language': language,
            'text': text_result_trim
        }
    except Exception as e:
        # Delete tmp file in case of error
        await async_remove(tmp_path)
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



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Set the maximum number of concurrent requests
max_concurrent_requests = 10
request_semaphore = Semaphore(max_concurrent_requests)


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



if __name__ == "__main__":
    # Start the FastAPI server
    logging.info("[Speech REST API] Starting server...")
    uvicorn.run("app_ex:app", host="0.0.0.0", port=3000, log_level="info")

