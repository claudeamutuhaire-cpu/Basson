FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt.
RUN pip install -r requirements.txt
# Install Piper
RUN apt-get update && apt-get install -y wget && \
    wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz && \
    tar -xzf piper_amd64.tar.gz && mv piper /usr/local/bin/
COPY app/.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]