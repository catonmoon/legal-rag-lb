FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy eval module
COPY scorer.py judge.py app.py ./

# Copy data files (docs + gold)
COPY gold_answers_final.json questions_final.json ./
COPY docs_final_md_fast/ ./docs_final_md_fast/
COPY docs_final/ ./docs_final/

EXPOSE 7861

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
