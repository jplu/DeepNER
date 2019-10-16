# coding=utf-8

import collections
import pandas as pd
import tensorflow as tf
import _pickle as pickle
import numpy as np
from absl import logging

vocab_tokens = ['[PAD]', '[UKN]']
vocab_chars = ['[PAD]', '[UKN]']


def _load_dataset(name):
    label_list = []
    dataset = {"text": [], "labels": []}
    logging.info(name + ": " + str(tf.io.gfile.exists(name)))
    with tf.io.gfile.GFile(name) as f:
        words = []
        labels = []
        for line in f:
            contents = line.strip()
            tokens = contents.split(u' ')

            if contents.startswith("-DOCSTART-"):
                continue
            if len(tokens) > 1:
                if tokens[0] not in vocab_tokens:
                    vocab_tokens.append(tokens[0])
                    for char in tokens[0]:
                        if char not in vocab_chars:
                            vocab_chars.append(char)
                words.append(tokens[0])
                labels.append(tokens[-1])
            else:
                if len(contents) == 0 and len(words) > 0:
                    for l in labels:
                        if l not in label_list:
                            label_list.append(l)
                    dataset["text"].append(' '.join(words))
                    dataset["labels"].append(' '.join(labels))
                    words = []
                    labels = []

    return pd.DataFrame.from_dict(dataset), label_list


class InputExample(object):
    def __init__(self, text=None, labels=None):
        self.text = text
        self.labels = labels


class InputFeatures(object):
    def __init__(self, chars, tokens, labels):
        self.chars = chars
        self.tokens = tokens
        self.labels = labels


def load_examples(conll_file):
    dataset, label_list = _load_dataset(conll_file)
    dataset_df = pd.concat([dataset]).sample(frac=1).reset_index(drop=True)

    dataset_examples = dataset_df.apply(
        lambda x: InputExample(text=x["text"], labels=x["labels"]), axis=1)

    return dataset_examples, label_list


def convert_tokens_to_ids(tokens):
    tokens_id = []

    for token in tokens:
        if token in vocab_tokens:
            tokens_id.append(vocab_tokens.index(token))
        else:
            tokens_id.append(vocab_tokens.index('[UKN]'))

    return tokens_id


def convert_chars_to_ids(chars):
    chars_id = []

    for char in chars:
        if char in vocab_chars:
            chars_id.append(vocab_chars.index(char))
        else:
            chars_id.append(vocab_chars.index('[UKN]'))

    return chars_id


def get_embedding_map(embeddings_path):
    embeddings = {}
    embedding_size = 0
    logging.info(embeddings_path + ": " + str(tf.io.gfile.exists(embeddings_path)))
    with tf.io.gfile.GFile(embeddings_path) as fp:
        for idx, line in enumerate(fp):
            if len(line) < 30 and idx == 0:
                # In fasttext, the first line is the # of vocab words.
                continue
            tokens = line.strip().split(u' ')
            word = tokens[0]
            embedding_size = len(tokens[1:])
            vec = list(map(float, tokens[1:]))

            if word not in vocab_tokens:
                continue
            embeddings[word] = vec
    embeddings['[PAD]'] = np.zeros(embedding_size).tolist()
    boundary = np.sqrt(3.0 / embedding_size)
    embeddings['[UKN]'] = np.random.uniform(-boundary, boundary, embedding_size).tolist()

    return list(embeddings.keys()), list(embeddings.values())


def convert_single_example(ex_index, example, label_list, max_token_seq_length,
                           max_sentence_seq_length):
    label_map = {}

    for (i, label) in enumerate(label_list):
        label_map[label] = i

    ntokens = example.text.strip().split(u' ')
    nlabels = example.labels.strip().split(u' ')
    tokens = [w.encode('utf-8') if w in vocab_tokens else b'[UKN]' for w in ntokens]
    labels = [l.encode('utf-8') for l in nlabels]
    chars = [[c.encode('utf-8') if c in vocab_chars else b'[UKN]' for c in w] for w in
             example.text.strip().split(u' ')]
    
    if len(tokens) > max_sentence_seq_length:
        tokens = tokens[0:max_sentence_seq_length]
        labels = labels[0:max_sentence_seq_length]
        ntokens = ntokens[0:max_sentence_seq_length]
        nlabels = nlabels[0:max_sentence_seq_length]
        chars = chars[0:max_sentence_seq_length]

    for i, _ in enumerate(chars):
        if len(chars[i]) > max_token_seq_length:
            chars[i] = chars[i][0:max_token_seq_length]
    
    if len(tokens) < max_sentence_seq_length:
        tokens.extend([b'[PAD]'] * (max_sentence_seq_length - len(tokens)))
        labels.extend([b'O'] * (max_sentence_seq_length - len(labels)))
        ntokens.extend(['[PAD]'] * (max_sentence_seq_length - len(ntokens)))
        nlabels.extend(['O'] * (max_sentence_seq_length - len(nlabels)))

    lengths = [len(c) for c in chars]
    chars = [c + [b'[PAD]'] * (max_token_seq_length - l) for c, l in zip(chars, lengths)]
    
    while len(chars) < max_sentence_seq_length:
        chars.append([b'[PAD]'] * max_token_seq_length)
    
    assert len(chars) == len(tokens) == len(labels) == len(ntokens) == len(nlabels) == \
           max_sentence_seq_length
    
    for tmp_chars in chars:
        assert len(tmp_chars) == max_token_seq_length

    chars = np.reshape([tmp_chars for tmp_chars in chars], -1).tolist()

    assert len(tokens) == max_sentence_seq_length
    assert len(chars) == max_sentence_seq_length * max_token_seq_length
    assert len(labels) == max_sentence_seq_length

    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        logging.info("chars: %s" % " ".join([str(x) for x in chars]))
        logging.info("labels: %s" % " ".join([str(x) for x in labels]))

    feature = InputFeatures(
        chars=chars,
        tokens=tokens,
        labels=labels,
    )

    return feature, ntokens, nlabels


def file_based_convert_examples_to_features(examples, label_list, max_token_seq_length,
                                            max_sentence_seq_length, output_file):
    writer = tf.io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature, tokens, labels = convert_single_example(ex_index, example, label_list,
                                                         max_token_seq_length,
                                                         max_sentence_seq_length)
        batch_tokens.extend(tokens)
        batch_labels.extend(labels)

        def create_bytes_feature(values):
            f = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))
            return f

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
            return f

        features = collections.OrderedDict()
        features["tokens"] = create_bytes_feature(feature.tokens)
        features["chars"] = create_bytes_feature(feature.chars)
        features["labels"] = create_bytes_feature(feature.labels)
        features["size_tokens"] = create_int_feature(np.full(max_sentence_seq_length,
                                                             max_token_seq_length))
        features["size_sentence"] = create_int_feature([max_sentence_seq_length])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()

    return batch_tokens, batch_labels


def create_features(max_token_seq_length, max_sentence_seq_length, train_conll, test_conll,
                    train_tfrecord_file, eval_tfrecord_file, metadata_file, embedding_path):
    global vocab_tokens

    if tf.io.gfile.exists(train_tfrecord_file):
        tf.io.gfile.remove(train_tfrecord_file)

    if tf.io.gfile.exists(eval_tfrecord_file):
        tf.io.gfile.remove(eval_tfrecord_file)

    if tf.io.gfile.exists(metadata_file):
        tf.io.gfile.remove(metadata_file)

    train_input_examples, label_list = load_examples(train_conll)
    eval_input_examples, _ = load_examples(test_conll)

    vocab_tokens, embeddings = get_embedding_map(embedding_path)
    _, _ = file_based_convert_examples_to_features(train_input_examples, label_list,
                                                   max_token_seq_length, max_sentence_seq_length,
                                                   train_tfrecord_file)
    batch_tokens, batch_labels = file_based_convert_examples_to_features(eval_input_examples,
                                                                         label_list,
                                                                         max_token_seq_length,
                                                                         max_sentence_seq_length,
                                                                         eval_tfrecord_file)

    metadata = {"max_token_seq_length": max_token_seq_length,
                "max_sentence_seq_length": max_sentence_seq_length, "labels": label_list,
                "train_number_examples": len(train_input_examples),
                "eval_number_examples": len(eval_input_examples), "embeddings": embeddings,
                "vocab_tokens": vocab_tokens, "vocab_chars": vocab_chars,
                "batch_tokens": batch_tokens, "batch_labels": batch_labels}

    with tf.io.gfile.GFile(metadata_file, "w") as f:
        pickle.dump(metadata, f)


def main():
    logging.set_verbosity(logging.INFO)
    create_features(45, 128, "../train.conll", "../test.conll", "../train.tfrecord",
                    "../eval.tfrecord", "../metadata.pkl", "../cc.fr.300.short.vec")


if __name__ == "__main__":
    main()
