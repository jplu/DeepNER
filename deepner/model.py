# coding=utf-8

import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
from deepner import dataset
import _pickle as pickle
from six.moves import reduce

MAX_SENTENCE_SEQ_LENGTH = 0
MAX_TOKEN_SEQ_LENGTH = 0


def masked_conv1d_and_max(t, weights, filters, kernel_size):
    # Get shape and parameters
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.to_float(weights)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    # Apply convolution
    t_conv = tf.layers.conv1d(t, filters, kernel_size, padding='same')
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
    t_max = tf.reduce_max(t_conv, axis=-2)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims-2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape)

    return t_max


def crf_loss(logits, labels, num_labels, mask2len):
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable("transition", shape=[num_labels, num_labels],
                                initializer=tf.contrib.layers.xavier_initializer())

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels,
                                                                   transition_params=trans,
                                                                   sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)

    return loss, transition


def create_model(is_training, char_ids, token_ids, label_ids, size_tokens, size_sentence,
                 num_labels, dropout, filters, kernel_size, lstm_size, vocab_word_embeddings,
                 total_vocab_chars):
    # Char Embeddings
    char_variable = tf.get_variable('chars_embeddings', [total_vocab_chars, 100], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(char_variable, char_ids)
    char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout, training=is_training)
    # Char 1d convolution
    weights = tf.sequence_mask(size_tokens)
    char_embeddings = masked_conv1d_and_max(char_embeddings, weights, filters, kernel_size)
    # Word Embeddings
    token_variable = tf.constant(np.asarray(vocab_word_embeddings), dtype=tf.float32)
    token_embeddings = tf.nn.embedding_lookup(token_variable, token_ids)
    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([token_embeddings, char_embeddings], axis=-1)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=is_training)
    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(lstm_size)
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(lstm_size)
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=size_sentence)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=size_sentence)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=is_training)
    logits = tf.layers.dense(output, num_labels)
    logits = tf.reshape(logits, [-1, MAX_SENTENCE_SEQ_LENGTH, num_labels])
    # CRF
    loss, trans = crf_loss(logits, label_ids, num_labels, size_sentence)
    predict, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, size_sentence)

    return loss, logits, predict


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, vocab_chars, vocab_tokens, vocab_labels, vocab_word_embeddings,
                     dropout, filters, kernel_size, lstm_size):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        features["chars"] = tf.reshape(features["chars"], [-1, MAX_SENTENCE_SEQ_LENGTH,
                                                           MAX_TOKEN_SEQ_LENGTH])

        if features["chars"].dtype == tf.string:
            lookup_tokens = tf.contrib.lookup.index_table_from_tensor(vocab_tokens, default_value=0)
            lookup_chars = tf.contrib.lookup.index_table_from_tensor(vocab_chars, default_value=0)
            lookup_labels = tf.contrib.lookup.index_table_from_tensor(vocab_labels, default_value=0)
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            char_ids = lookup_chars.lookup(features["chars"])
            token_ids = lookup_tokens.lookup(features["tokens"])
            label_ids = lookup_labels.lookup(features["labels"])
        else:
            char_ids = features["chars"]
            token_ids = features["tokens"]
            label_ids = features["labels"]

        total_loss, logits, preds = create_model(is_training, char_ids, token_ids, label_ids,
                                                 features["size_tokens"], features["size_sentence"],
                                                 num_labels, dropout, filters, kernel_size,
                                                 lstm_size, vocab_word_embeddings, len(vocab_chars))

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                total_loss, global_step=tf.train.get_or_create_global_step())
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=total_loss,
                                                          train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits, num_labels, mask):
                predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)

                return {
                    "confusion_matrix": _streaming_confusion_matrix(label_ids, predictions,
                                                                    num_labels-1, weights=mask)
                }
            mask = tf.sequence_mask(features["size_sentence"])
            eval_metrics = (metric_fn, [label_ids, logits, num_labels, mask])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=total_loss,
                                                          eval_metrics=eval_metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=preds)

        return output_spec

    # Return the actual model function in the closure
    return model_fn


def serving_input_fn():
    tokens = tf.placeholder(tf.string, [None, MAX_SENTENCE_SEQ_LENGTH], name='tokens')
    labels = tf.placeholder(tf.string, [None, MAX_SENTENCE_SEQ_LENGTH], name='labels')
    chars = tf.placeholder(tf.string, [None, MAX_SENTENCE_SEQ_LENGTH, MAX_TOKEN_SEQ_LENGTH],
                           name='chars')
    size_tokens = tf.placeholder(tf.int32, [None, MAX_SENTENCE_SEQ_LENGTH], name='size_tokens')
    size_sentence = tf.placeholder(tf.int32, [None], name='size_sentence')

    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'tokens': tokens,
        'chars': chars,
        'labels': labels,
        'size_tokens': size_tokens,
        'size_sentence': size_sentence,
    })()

    return input_fn


def file_based_input_fn_builder(input_file, is_training, drop_remainder):
    name_to_features = {
        "tokens": tf.FixedLenFeature([MAX_SENTENCE_SEQ_LENGTH], tf.string),
        "chars": tf.FixedLenFeature([MAX_SENTENCE_SEQ_LENGTH * MAX_TOKEN_SEQ_LENGTH], tf.string),
        "labels": tf.FixedLenFeature([MAX_SENTENCE_SEQ_LENGTH], tf.string),
        "size_tokens": tf.FixedLenFeature([MAX_SENTENCE_SEQ_LENGTH], tf.int64),
        "size_sentence": tf.FixedLenFeature([1], tf.int64),
    }

    def _decode_record(record):
        example = tf.parse_single_example(record, name_to_features)
        example["size_sentence"] = tf.constant(MAX_SENTENCE_SEQ_LENGTH, dtype=tf.int32)

        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record), batch_size=batch_size,
            num_parallel_calls=8, drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)

        return d

    return input_fn


def write_results(output_predict_file, result, batch_tokens, batch_labels, label_list):
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        predictions = []

        for _, pred in enumerate(result):
            predictions.extend(pred)

        for i, prediction in enumerate(predictions):
            token = batch_tokens[i]
            predict = label_list[prediction]
            true_label = batch_labels[i]

            if token != "[PAD]":
                line = "{}\t{}\t{}\n".format(token, predict, true_label)
                writer.write(line)


def calculate(total_cm, num_class):
    precisions = []
    recalls = []
    f1s = []

    for i in range(num_class):
        rowsum, colsum = np.sum(total_cm[i]), np.sum(total_cm[r][i] for r in range(num_class))
        precision = total_cm[i][i] / float(colsum + 1e-12)
        recall = total_cm[i][i] / float(rowsum + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)


def train(config):
    global MAX_SENTENCE_SEQ_LENGTH
    global MAX_TOKEN_SEQ_LENGTH
    MAX_SENTENCE_SEQ_LENGTH = config['max_sentence_seq_length']
    MAX_TOKEN_SEQ_LENGTH = config['max_token_seq_length']
    train_conll = os.path.join(config['data_dir'], "train.conll")
    test_conll = os.path.join(config['data_dir'], "test.conll")
    train_tfrecord_file = os.path.join(config['data_dir'], "train.tfrecord")
    eval_tfrecord_file = os.path.join(config['data_dir'], "eval.tfrecord")
    metadata_file = os.path.join(config['data_dir'], "metadata.pkl")
    output_dir = config['output_dir']
    embedding_path = os.path.join(config['data_dir'], config['embedding_path'])

    if not tf.gfile.Exists(train_tfrecord_file) or not tf.gfile.Exists(eval_tfrecord_file) \
            or not tf.gfile.Exists(metadata_file):
        dataset.create_features(MAX_TOKEN_SEQ_LENGTH, MAX_SENTENCE_SEQ_LENGTH, train_conll,
                                test_conll, train_tfrecord_file, eval_tfrecord_file, metadata_file,
                                embedding_path)

    with tf.gfile.GFile(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    label_list = metadata["labels"]
    train_number_examples = metadata["train_number_examples"]
    predict_eval_number_examples = metadata["eval_number_examples"]
    embeddings = metadata["embeddings"]
    vocab_chars = metadata["vocab_chars"]
    vocab_tokens = metadata["vocab_tokens"]
    batch_tokens = metadata["batch_tokens"]
    batch_labels = metadata["batch_labels"]
    tpu_cluster_resolver = None

    if config['use_tpu']:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            config['tpu_name'], zone=config['tpu_zone'], project=config['gcp_project'])

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(cluster=tpu_cluster_resolver, master=config['master'],
                                          model_dir=output_dir,
                                          save_checkpoints_steps=config['save_checkpoints_steps'],
                                          tpu_config=tf.contrib.tpu.TPUConfig(
                                              iterations_per_loop=config['iterations_per_loop'],
                                              num_shards=config['num_tpu_cores'],
                                              per_host_input_for_training=is_per_host))

    num_train_steps = int(train_number_examples / config['train_batch_size'] *
                          config['num_train_epochs'])

    model_fn = model_fn_builder(num_labels=len(label_list) + 1, vocab_chars=vocab_chars,
                                vocab_tokens=vocab_tokens, vocab_labels=label_list,
                                vocab_word_embeddings=embeddings, dropout=config['dropout'],
                                filters=config['filters'], kernel_size=config['kernel_size'],
                                lstm_size=config['lstm_size'])

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=config['use_tpu'], model_fn=model_fn,
                                            config=run_config,
                                            train_batch_size=config['train_batch_size'],
                                            eval_batch_size=config['eval_batch_size'],
                                            predict_batch_size=config['predict_batch_size'])
    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = file_based_input_fn_builder(input_file=train_tfrecord_file,
                                                 is_training=True,
                                                 drop_remainder=True)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", train_number_examples)
    tf.logging.info("  Batch size = %d", config['train_batch_size'])
    tf.logging.info("  Num steps = %d", num_train_steps)

    current_time = datetime.datetime.now()

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    tf.logging.info("  Training took time = {}".format(datetime.datetime.now() - current_time))

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if config['use_tpu']:
        # Eval will be slightly WRONG on the TPU because it will truncate
        # the last batch.
        assert len(predict_eval_number_examples) % config['eval_batch_size'] == 0
        eval_steps = int(len(predict_eval_number_examples) // config['eval_batch_size'])

    eval_drop_remainder = True if config['use_tpu'] else False
    eval_predict_input_fn = file_based_input_fn_builder(input_file=eval_tfrecord_file,
                                                        is_training=False,
                                                        drop_remainder=eval_drop_remainder)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", predict_eval_number_examples)
    tf.logging.info("  Batch size = %d", config['eval_batch_size'])

    result = estimator.evaluate(input_fn=eval_predict_input_fn, steps=eval_steps)
    output_eval_file = os.path.join(output_dir, "eval_results.txt")

    with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")

        confusion_matrix = result["confusion_matrix"]
        precision, recall, f1 = calculate(confusion_matrix, len(label_list) - 1)

        tf.logging.info("  precision = %s", str(precision))
        tf.logging.info("  recall = %s", str(recall))
        tf.logging.info("  f1 = %s", str(f1))

        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    if config['use_tpu']:
        assert len(predict_eval_number_examples) % config['predict_batch_size'] == 0

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", predict_eval_number_examples)
    tf.logging.info("  Batch size = %d", config['predict_batch_size'])

    result = estimator.predict(input_fn=eval_predict_input_fn)
    output_predict_file = os.path.join(output_dir, "test_results.tsv")
    write_results(output_predict_file, result, batch_tokens, batch_labels, label_list)

    estimator._export_to_tpu = False

    estimator.export_savedmodel(output_dir, serving_input_fn)
