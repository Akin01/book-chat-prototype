FROM python:3.10-slim-bullseye
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

CMD [ "streamlit", "run", "app.py" ]