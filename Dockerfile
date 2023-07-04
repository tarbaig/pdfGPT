FROM python:3.9-slim-buster as langchain-serve-img

RUN apt update && apt install -y gcc

RUN pip3 install uvicorn
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install python-multipart
COPY . .
CMD [ "uvicorn", "api:app", "--log-level", "info"]

FROM python:3.9-slim-buster as pdf-gpt-img
ENV OPENAI_API_KEY='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'
RUN apt update && apt install -y gcc

WORKDIR /app
COPY requirements-2.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "app.py" ]
