FROM python:3.11-bookworm

ENV U2NET_HOME=/workspace/.u2net

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt \
    && mkdir -p "${U2NET_HOME}" \
    && python -c "from rembg.session_factory import new_session; new_session('u2net')"

EXPOSE 8080

CMD ["python3", "app.py", "--log-level=DEBUG"]
