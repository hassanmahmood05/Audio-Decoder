FROM python:3.12-slim

# ffmpeg/ffprobe: decoding and validation of every input format
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Optional: bake the model into the image so containers start without a download.
ARG WHISPER_MODEL=small
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

ENV AUDIO_DECODER_WHISPER_MODEL=${WHISPER_MODEL} \
    AUDIO_DECODER_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "audio_decoder.api:app", "--host", "0.0.0.0", "--port", "8000"]
