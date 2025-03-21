FROM python:3.11
ADD . /code
WORKDIR /code
RUN apt-get update && apt-get install -y build-essential ffmpeg libspeex-dev libspeexdsp-dev libpulse-dev && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
CMD exec gunicorn -k gevent -b 0.0.0.0:$PORT asr:app