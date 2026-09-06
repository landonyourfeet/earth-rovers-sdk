FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies first for layer caching, then the browsers.
#   - Playwright's Chromium: always present, no proprietary codecs.
#   - Google Chrome (via Playwright's "chrome" channel): ships H.264/AAC, which
#     the rover's video stream prefers. browser_service.py picks Chrome when it
#     is installed and falls back to Chromium otherwise.
# The Chrome install is NOT allowed to fail silently: a build that ships without
# it would come up "successful" and blind, so the step fails the build instead.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
  && playwright install --with-deps chromium \
  && playwright install --with-deps chrome \
  && (google-chrome --version || /opt/google/chrome/chrome --version) \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python3", "-m", "hypercorn", "main:app", "--bind", "0.0.0.0:8000"]
