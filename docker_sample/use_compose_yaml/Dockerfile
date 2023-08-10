FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y && \
    rm -rf /var/lib/apt/lists/*

COPY ./run.sh /app/run.sh
WORKDIR /app

CMD ["bash", "run.sh"]
