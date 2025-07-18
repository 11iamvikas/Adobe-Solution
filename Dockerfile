FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Copy the processing script
COPY process_pdfs.py .
# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run the script
CMD ["python", "process_pdfs.py"] 