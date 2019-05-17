# coding=utf-8

import tensorflow as tf
from deepner.model import train

tf.flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .conll files (or other data files) "
    "for the task.")

tf.flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

tf.flags.DEFINE_string(
    "embedding_path", "glove.840B.300d.txt",
    "The glove or fastText embedding file.")

tf.flags.DEFINE_integer(
    "max_sentence_seq_length", 128,
    "The maximum total input sentence length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

tf.flags.DEFINE_integer(
    "max_token_seq_length", 45,
    "The maximum total input token length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

tf.flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

tf.flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

tf.flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")

tf.flags.DEFINE_integer("iterations_per_loop", 1000,
                        "How many steps to make in each estimator call.")

tf.flags.DEFINE_integer("train_batch_size", 128, "Total batch size for training.")

tf.flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

tf.flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

tf.flags.DEFINE_integer("kernel_size", 3, "Kernel size for the characters convolution.")

tf.flags.DEFINE_integer("filters", 50, "Filters for the characters convolution.")

tf.flags.DEFINE_integer("lstm_size", 100, "Size for the LSTM layers.")

tf.flags.DEFINE_float("dropout", 0.5, "The dropout used during the training.")

tf.flags.DEFINE_float("num_train_epochs", 50.0, "Total number of training epochs to perform.")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if tf.flags.FLAGS.use_tpu and not tf.flags.FLAGS.tfhub_cache_dir:
        raise ValueError("The option --tfhub_cache_dir must be set if TPU is used")

    train(tf.flags.FLAGS.flag_values_dict())


if __name__ == "__main__":
    tf.flags.mark_flag_as_required("data_dir")
    tf.flags.mark_flag_as_required("output_dir")
    tf.app.run()
