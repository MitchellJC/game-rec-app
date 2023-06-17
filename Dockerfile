# syntax=docker/dockerfile:1

FROM python:latest

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN python data_init.py
CMD ["python", "-m" , "flask", "run", "--host=0.0.0.0"]
