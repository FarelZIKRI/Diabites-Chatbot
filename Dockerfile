# Gunakan base image Python resmi yang ringan
FROM python:3.12-slim

# Set environment variables untuk efisiensi
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Buat working directory
WORKDIR /code

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements dan install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Salin seluruh kode proyek ke dalam container
COPY . /code

# Buat folder logs agar tidak terjadi error runtime
RUN mkdir -p /code/logs

# Port default yang diwajibkan oleh Hugging Face
EXPOSE 7860

# Jalankan aplikasi menggunakan Uvicorn di port 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
