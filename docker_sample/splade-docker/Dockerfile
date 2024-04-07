FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN apt-get update
RUN apt-get install git -y

WORKDIR /app
RUN git clone https://github.com/wararaki718/splade.git
WORKDIR /app/splade

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /app/splade/checkpoint /app/splade/model_checkpoint  /app/splade/output
# ENTRYPOINT [ "python", "main.py" ]
