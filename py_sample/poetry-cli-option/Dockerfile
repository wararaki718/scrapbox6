FROM python:3.8

RUN pip install poetry==1.3.2

COPY . /opt/app
WORKDIR /opt/app
RUN poetry install

ENTRYPOINT [ "poetry", "run", "cliapp" ]
