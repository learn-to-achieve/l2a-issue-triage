FROM python:3.11-slim

WORKDIR /app

# Board-only deps (streamlit + pandas) — NOT the pipeline's ML stack.
COPY requirements-board.txt .
RUN pip install --no-cache-dir -r requirements-board.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# Shell form so ${PORT} expands; Cloud Run injects PORT (default 8080).
CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
