FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY smart_pdf_processor/ smart_pdf_processor/

# Download NLTK data to a known location
RUN mkdir -p /usr/share/nltk_data && \
    python3 -m nltk.downloader -d /usr/share/nltk_data stopwords punkt

# Set environment variable so NLTK always finds the data
ENV NLTK_DATA=/usr/share/nltk_data

# Set default command
CMD ["python3", "smart_pdf_processor/process.py"]
