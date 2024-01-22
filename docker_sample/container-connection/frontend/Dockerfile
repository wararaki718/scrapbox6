FROM python:3.10

RUN pip install fastapi uvicorn requests

COPY ./frontend /opt/app
WORKDIR /opt/app

ENTRYPOINT [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1234" ]
