FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them on every build.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as a regular user. UID 1000 matches the default user on most Linux servers (including
# the "ubuntu" user on AWS), so files written to the mounted data/ folder stay editable.
RUN useradd --uid 1000 --create-home app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
