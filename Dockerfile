FROM python:3.11-bookworm

# Set environment variables for model caching
ENV U2NET_HOME=/workspace/.u2net
ENV SENTENCE_TRANSFORMERS_HOME=/workspace/.cache/sentence_transformers
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

WORKDIR /workspace

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Pre-download ML models during build to avoid runtime downloads
RUN echo "Downloading sentence-transformers model..." \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && echo "Model downloaded successfully"

# Copy application code
COPY . .

# Download rembg model
RUN mkdir -p "${U2NET_HOME}" \
    && python -c "from rembg.session_factory import new_session; new_session('u2net')"

EXPOSE 8080

CMD ["python3", "app.py", "--log-level=DEBUG"]
