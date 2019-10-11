# coding=utf-8
import grpc
import os
from fastapi import FastAPI
from pydantic import BaseModel
import _pickle as pickle
import tensorflow as tf
from spacy.tokens import Span
from spacy.lang.en import English
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from starlette.middleware.cors import CORSMiddleware


class InputFeatures(object):
    def __init__(self, chars, tokens, labels):
        self.chars = chars
        self.tokens = tokens
        self.labels = labels


class InputExample(object):
    def __init__(self, text=None, labels=None):
        self.text = text
        self.labels = labels


class Ad(BaseModel):
    text: str


# tf.logging.set_verbosity(tf.logging.INFO)

app = FastAPI()

# CORS
origins = ["*"]

app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

METADATA_FILE = os.environ.get("METADATA_FILE")
MODEL_NAME = os.environ.get("MODEL_NAME")
HOST_NAME = os.environ.get("HOST_NAME") if os.environ.get("HOST_NAME") else "localhost"

with tf.io.gfile.GFile(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

label_list = metadata["labels"]
MAX_SENTENCE_SEQ_LENGTH = metadata['max_sentence_seq_length']
MAX_TOKEN_SEQ_LENGTH = metadata['max_token_seq_length']
vocab_tokens = metadata["vocab_tokens"]
vocab_chars = metadata["vocab_chars"]
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
channel = grpc.insecure_channel(HOST_NAME + ":8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def convert_single_example(ex_index, example):
    label_map = {}

    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens = [w.encode('utf-8') if w in vocab_tokens else b'[UKN]' for w in
              example.text.strip().split(u' ')]
    labels = [l.encode('utf-8') for l in example.labels.strip().split(u' ')]

    if len(tokens) > MAX_SENTENCE_SEQ_LENGTH:
        tokens = tokens[0:MAX_SENTENCE_SEQ_LENGTH]
        labels = labels[0:MAX_SENTENCE_SEQ_LENGTH]

    chars = [[c.encode('utf-8') if c in vocab_chars else b'[UKN]' for c in w] for w in
             example.text.strip().split(u' ')]

    for i, _ in enumerate(chars):
        if len(chars[i]) > MAX_TOKEN_SEQ_LENGTH:
            chars[i] = chars[i][0:MAX_TOKEN_SEQ_LENGTH]

    if len(tokens) < MAX_SENTENCE_SEQ_LENGTH:
        tokens.extend([b'[PAD]'] * (MAX_SENTENCE_SEQ_LENGTH - len(tokens)))
        labels.extend([b'O'] * (MAX_SENTENCE_SEQ_LENGTH - len(labels)))

    lengths = [len(c) for c in chars]
    chars = [c + [b'[PAD]'] * (MAX_TOKEN_SEQ_LENGTH - l) for c, l in zip(chars, lengths)]

    while len(chars) < MAX_SENTENCE_SEQ_LENGTH:
        chars.append([b'[PAD]'] * MAX_TOKEN_SEQ_LENGTH)

    for tmp_chars in chars:
        assert len(tmp_chars) == MAX_TOKEN_SEQ_LENGTH

    assert len(tokens) == MAX_SENTENCE_SEQ_LENGTH
    assert len(chars) == MAX_SENTENCE_SEQ_LENGTH
    assert len(labels) == MAX_SENTENCE_SEQ_LENGTH

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        tf.logging.info("chars: %s" % " ".join([str(x) for x in chars]))
        tf.logging.info("labels: %s" % " ".join([str(x) for x in labels]))

    feature = InputFeatures(
        chars=chars,
        tokens=tokens,
        labels=labels,
    )

    return feature


@app.get("/api/ner/health")
async def health():
    return {"message": "I'm alive"}


@app.post("/api/ner/recognize")
async def get_prediction(ad: Ad):
    doc = tokenizer(ad.text)
    txt = " ".join([token.text for token in doc])
    input_example = InputExample(text=txt, labels=" ".join(['O'] * len(txt)))
    feature = convert_single_example(0, input_example)
    model_request = predict_pb2.PredictRequest()
    model_request.model_spec.name = MODEL_NAME
    model_request.model_spec.signature_name = 'serving_default'
    tokens = tf.contrib.util.make_tensor_proto(feature.tokens,
                                               shape=[1, MAX_SENTENCE_SEQ_LENGTH])
    chars = tf.contrib.util.make_tensor_proto(feature.chars,
                                              shape=[1, MAX_SENTENCE_SEQ_LENGTH,
                                                     MAX_TOKEN_SEQ_LENGTH])
    labels = tf.contrib.util.make_tensor_proto(feature.labels, shape=[1, MAX_SENTENCE_SEQ_LENGTH])
    size_tokens = tf.contrib.util.make_tensor_proto([MAX_TOKEN_SEQ_LENGTH] *
                                                    MAX_SENTENCE_SEQ_LENGTH,
                                                    shape=[1, MAX_SENTENCE_SEQ_LENGTH])
    size_sentence = tf.contrib.util.make_tensor_proto(MAX_SENTENCE_SEQ_LENGTH, shape=[1])
    model_request.inputs['tokens'].CopyFrom(tokens)
    model_request.inputs['chars'].CopyFrom(chars)
    model_request.inputs['labels'].CopyFrom(labels)
    model_request.inputs['size_tokens'].CopyFrom(size_tokens)
    model_request.inputs['size_sentence'].CopyFrom(size_sentence)
    result = stub.Predict(model_request, 5.0)
    result = tf.make_ndarray(result.outputs["output"])

    return output(doc, result[0])


def output(doc, ids):
    res = {"entities": []}
    entities = []
    tf.logging.info(ids)
    tf.logging.info(label_list)
    annotations = ids[0:len(doc)]
    tf.logging.info(list(doc))
    tf.logging.info([label_list[label] for label in annotations])
    assert len(doc) == len(annotations)
    prev_type = label_list[annotations[0]]
    start_span = 0
    end_span = 1

    for idx in range(1, len(annotations)):
        if prev_type != label_list[annotations[idx]] and prev_type != 'O':
            entities.append({"type": prev_type, "start": start_span, "end": end_span})
            prev_type = label_list[annotations[idx]]
            start_span = idx
            end_span = idx + 1
        elif annotations[idx] != 'O' and prev_type != 'O':
            end_span += 1
        else:
            prev_type = label_list[annotations[idx]]
            start_span = idx
            end_span = idx + 1

    if prev_type != 'O':
        entities.append({"type": prev_type, "start": start_span, "end": end_span})

    for ent in entities:
        span = Span(doc, ent["start"], ent["end"], label=ent["type"])
        doc.ents = list(doc.ents) + [span]

    for ent in doc.ents:
        res["entities"].append({"phrase": ent.text, "type": ent.label_,
                                "startOffset": ent.start_char, "endOffset": ent.end_char})

    return res
