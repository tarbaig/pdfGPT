import logging
import os
import re
import shutil
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Dict

import fitz
import numpy as np
import openai
import tensorflow_hub as hub
from fastapi import UploadFile, Form

from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, Response

from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from twilio.rest import Client

app = FastAPI()

recommender = None
logger = logging.getLogger(__name__)

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    def __init__(self):
        self.use = hub.load('./USE')
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i : (i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def load_recommender(paths, start_page=1):
    global recommender
    if recommender is None:
        recommender = SemanticSearch()

    chunks : List[str] = []
    for p in paths:
        texts = pdf_to_text(p, start_page=start_page)
        p_chunks = text_to_chunks(texts, start_page=start_page)
        chunks.extend(p_chunks)
    recommender.fit(chunks)
    return 'Corpus Loaded.'


def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    try:
        completions = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=512,
            n=1,
            stop=None,
            temperature=0.7,
        )
        message = completions.choices[0].text
    except Exception as e:
        message = f'API Error: {str(e)}'
    return message 


def generate_answer(question, openAI_key):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "
        "Citation should be done at the end of each sentence. If the search results mention multiple subjects "
        "with the same name, create separate answers for each. Only include information found in the results and "
        "don't add any additional information. Make sure the answer is correct and don't output false content. "
        "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "
        "search results which has nothing to do with the question. Only answer what is asked. The "
        "answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "
    )

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(openAI_key, prompt, "text-davinci-003")
    return answer


def load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key is None:
        raise ValueError(
            "[ERROR]: Please pass your OPENAI_API_KEY. Get your key here : https://platform.openai.com/account/api-keys"
        )
    return key


@app.post("/ask_url")
def ask_url(url: str, question: str):
    download_pdf(url, 'corpus.pdf')
    load_recommender('corpus.pdf')
    openAI_key = load_openai_key()
    return generate_answer(question, openAI_key)


@app.post("/ask_file")
async def ask_file(files: List[UploadFile], question: str) -> str:

    temp_paths : List[Path] = []
    for file in files:
        suffix = Path(file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_paths.append(Path(tmp.name))

    load_recommender([str(tmp_path) for tmp_path in temp_paths])
    openAI_key = load_openai_key()
    return generate_answer(question, openAI_key)

@app.post("/whatsapp_enndpoint")
async def whats_app(
         From: str = Form(...), To: str = Form(...), Body: str = Form(...)  ):
    logger.error(From)
    logger.error(To)
    logger.error(Body)
    openAI_key = load_openai_key()
    ans = generate_answer(Body, openAI_key)
    logger.error(ans)
    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    client = Client(account_sid, auth_token)

    message = client.messages.create(
                              body=ans,
                              from_=To,
                              to=From
                          )

    print(message.sid)
