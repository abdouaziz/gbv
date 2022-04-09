FROM ubuntu:latest

COPY ./api /api
COPY requirements.txt /requirements.txt



RUN apt-get update \
    && apt-get install python3-pip -y \
    && pip3 install -r requirements.txt

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0"]