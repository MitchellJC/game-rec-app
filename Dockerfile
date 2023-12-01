# syntax=docker/dockerfile:1

FROM python:3.11

WORKDIR /app

COPY requirements.txt requirements.txt
COPY . .
RUN pip install -r requirements.txt

# CMD ["python", "-m" , "flask", "run", "--host=0.0.0.0", "--port", "5000"]
CMD ["python", "app.py"]
