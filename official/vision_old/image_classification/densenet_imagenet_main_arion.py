from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
import os, sys 
import time
sys.path.append('/tmp/arion/scalability/models-master')
from official.benchmark.models import trivial_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.vision.image_classification import common
from official.vision.image_classification import imagenet_preprocessing
# from official.vision.image_classification.models.resnet import ResNet101
# from official.vision.image_classification.models.inception_v3 import InceptionV3
# from official.vision.image_classification.models.vgg16 import VGG16
from arion.strategy.ps_strategy import PS
from arion.strategy.ps_lb_strategy import PSLoadBalancing
from arion.strategy.partitioned_ps_strategy import PartitionedPS
from arion.strategy.all_reduce_strategy import AllReduce
from arion.strategy.parallax_strategy import Parallax
from densenet_models import DenseNetS1, DenseNetS2, DenseNetS3, DenseNetS4, DenseNetS5, DenseNet121, DenseNet169, DenseNet201,\
    DenseNetS6, DenseNetS7, DenseNetS8, DenseNetS9, DenseNetS10, DenseNetS11, DenseNetS12, DenseNetS13, DenseNetS14, DenseNetS15, DenseNetS16,\
    DenseNetS17, DenseNetS18, DenseNetS19, DenseNetS20
#flags.DEFINE_integer(name='node_num', default=1, help='the number of nodes')
#flags.DEFINE_integer(name='gpu_num', default=1, help='the number of gpus of each node')
flags.DEFINE_string(name='cnn_model', default='resnet101', help='model to test')
flags.DEFINE_boolean(name='is_arion', default=False, help='is the arion mode enabled')
flags.DEFINE_string(name='arion_strategy', default='PS', help='the arion strategy')
flags.DEFINE_boolean(name='arion_patch_tf', default=True, help='ARION_PATCH_TF')
flags.DEFINE_integer(name='chunk_size', default=128, help='chunk size')

resource_spec_file = os.path.join(
    os.path.dirname(__file__),
    'resource_spec.yml')
import yaml
resource_info = yaml.safe_load(open(resource_spec_file, 'r'))
try:
  node_num = len(resource_info['nodes'])
except ValueError:
  print("nodes need to be set in specficiation file")

try:
  gpu_num = len(resource_info['nodes'][0]['gpus'])
except ValueError:
  print("gpus need to be set in specficiation file")

class TimeHistory(object):
  def __init__(self, batch_size, log_steps):
    self.batch_size = batch_size
    self.log_steps = log_steps ## typically the number of steps in each epoch
    self.global_steps = 0
    self.epoch_num = 0
    self.examples_per_second = 0
    logging.info("batch steps: %f", log_steps)

  def on_train_end(self):
    self.train_finish_time = time.time()
    elapsed_time = self.train_finish_time - self.train_start_time
    logging.info(
      "total time take: %f,"
      "averaged examples_per_second: %f",
      elapsed_time, self.examples_per_second / self.epoch_num)

  def on_epoch_begin(self, epoch):
    self.epoch_num += 1
    self.epoch_start = time.time()

  def on_batch_begin(self, batch):
    self.global_steps += 1
    if self.global_steps == 1:
      self.train_start_time = time.time()
      self.start_time = time.time()

  def on_batch_end(self, batch, loss):
    """Records elapse time of the batch and calculates examples per second."""
    logging.info(
      "global step:%d, loss value: %f",
      self.global_steps, loss)
    if self.global_steps % self.log_steps == 0:
      timestamp = time.time()
      elapsed_time = timestamp - self.start_time
      examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
      logging.info(
        "global step:%d, time_taken: %f,"
        "examples_per_second: %f",
          self.global_steps, elapsed_time, examples_per_second)
      self.examples_per_second += examples_per_second
      self.start_time = timestamp

  def on_epoch_end(self, epoch):
    epoch_run_time = time.time() - self.epoch_start
    logging.info(
      "epoch':%d, 'time_taken': %f",
      epoch, epoch_run_time)

def run(flags_obj):
  """Run ResNet ImageNet training and eval loop using native Keras APIs.

  Args:/storage/baoyu/arion/scalability/models-master/official/vision/image_classification/resnet_imagenet_main.py
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  # keras_utils.set_session_config(
  #     enable_eager=flags_obj.enable_eager,
  #     enable_xla=flags_obj.enable_xla)

  # Execute flag override logic for better model performance
  # if flags_obj.tf_gpu_thread_mode:
  #   keras_utils.set_gpu_thread_mode_and_count(
  #       per_gpu_thread_count=flags_obj.per_gpu_thread_count,
  #       gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
  #       num_gpus=flags_obj.num_gpus,
  #       datasets_num_private_threads=flags_obj.datasets_num_private_threads)
  # common.set_cudnn_batchnorm_mode()

  if flags_obj.arion_patch_tf:
    os.environ['ARION_PATCH_TF'] = '1'
  else:
    os.environ['ARION_PATCH_TF'] = '0'

  from arion import Arion
  config_file = os.path.join(os.path.dirname(__file__), 'runner_config.yml')

  if flags_obj.arion_strategy == 'PS':
    arion = Arion(resource_spec_file, PS(), runner_config_file=config_file)
  elif flags_obj.arion_strategy == 'PSLoadBalancing':
    arion = Arion(resource_spec_file, PSLoadBalancing(), runner_config_file=config_file)  
  elif flags_obj.arion_strategy == 'PartitionedPS':
    arion = Arion(resource_spec_file, PartitionedPS(), runner_config_file=config_file)    
  elif flags_obj.arion_strategy == 'AllReduce':
    arion = Arion(resource_spec_file, AllReduce(flags_obj.chunk_size), runner_config_file=config_file)    
  elif flags_obj.arion_strategy == 'Parallax':
    arion = Arion(resource_spec_file, Parallax(), runner_config_file=config_file)
  else:
    raise ValueError('the strategy can be only from PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax')
  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == tf.float16:
    loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
    policy = tf.compat.v1.keras.mixed_precision.experimental.Policy(
        'mixed_float16', loss_scale=loss_scale)
    tf.compat.v1.keras.mixed_precision.experimental.set_policy(policy)
    if not keras_utils.is_v2_0():
      raise ValueError('--dtype=fp16 is not supported in TensorFlow 1.')
  elif dtype == tf.bfloat16:
    policy = tf.compat.v1.keras.mixed_precision.experimental.Policy(
        'mixed_bfloat16')
    tf.compat.v1.keras.mixed_precision.experimental.set_policy(policy)

  data_format = flags_obj.data_format
  # if data_format is None:
  #   data_format = ('channels_first'
  #                  if tf.test.is_built_with_cuda() else 'channels_last')
  # tf.keras.backend.set_image_data_format(data_format)

  # Configures cluster spec for distribution strategy.
  # num_workers = distribution_utils.configure_cluster(flags_obj.worker_hosts,
  #                                                    flags_obj.task_index)

  # strategy = distribution_utils.get_distribution_strategy(
  #     distribution_strategy=flags_obj.distribution_strategy,
  #     num_gpus=flags_obj.num_gpus,
  #     num_workers=num_workers,
  #     all_reduce_alg=flags_obj.all_reduce_alg,
  #     num_packs=flags_obj.num_packs,
  #     tpu_address=flags_obj.tpu)

  # if strategy:
  #   # flags_obj.enable_get_next_as_optional controls whether enabling
  #   # get_next_as_optional behavior in DistributedIterator. If true, last
  #   # partial batch can be supported.
  #   strategy.extended.experimental_enable_get_next_as_optional = (
  #       flags_obj.enable_get_next_as_optional
  #   )

  # strategy_scope = distribution_utils.get_strategy_scope(strategy)

  # pylint: disable=protected-access
  # if flags_obj.use_synthetic_data:
  #   distribution_utils.set_up_synthetic_data()
  #   input_fn = common.get_synth_input_fn(
  #       height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
  #       width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
  #       num_channels=imagenet_preprocessing.NUM_CHANNELS,
  #       num_classes=imagenet_preprocessing.NUM_CLASSES,
  #       dtype=dtype,  # if flags_obj.skip_eval:
  #   # Only build the training graph. This reduces memory usage introduced by
  #   # control flow ops in layers that have different implementations for
  #   # training and inference (e.g., batch norm).
  #   if flags_obj.set_learning_phase_to_train:
  #     # TODO(haoyuzhang): Understand slowdown of setting learning phase when
  #     # not using distribution strategy.
  #     tf.keras.backend.set_learning_phase(1)
  #       drop_remainder=True)
  # else:
  #   distribution_utils.undo_set_up_synthetic_data()
  input_fn = imagenet_preprocessing.input_fn

  # When `enable_xla` is True, we always drop the remainder of the batches
  # in the dataset, as XLA-GPU doesn't support dynamic shapes.
  drop_remainder = flags_obj.enable_xla



  ###########TO DO: Evaluation###########
  # eval_input_dataset = None
  # if not flags_obj.skip_eval:
  #   eval_input_dataset = input_fn(
  #       is_training=False,
  #       data_dir=flags_obj.data_dir,https://stackoverflow.com/questions/15474095/writing-a-log-file-from-python-program
  #       batch_size=flags_obj.batch_size,
  #       num_epochs=flags_obj.train_epochs,
  #       parse_record_fn=imagenet_preprocessing.parse_record,
  #       dtype=dtype,
  #       drop_remainder=drop_remainder)
  if flags_obj.cnn_model == 'vgg16':
    lr_schedule = 0.01  
  else:
    lr_schedule = 0.1
  if flags_obj.use_tensor_lr:
    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=flags_obj.batch_size,
        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True)

  #with strategy_scope:
  #with tf.Graph().as_default(), arion.scope():
  with tf.Graph().as_default(), arion.scope() if flags_obj.is_arion else tf.Graph().as_default():
    #optimizer = common.get_optimizer(lr_schedule)

    train_input_dataset = input_fn(
        is_training=True,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        num_epochs=flags_obj.train_epochs,
        parse_record_fn=imagenet_preprocessing.parse_record,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads,
        dtype=dtype,
        drop_remainder=drop_remainder,
        tf_data_experimental_slack=flags_obj.tf_data_experimental_slack,
        training_dataset_cache=flags_obj.training_dataset_cache,
    )

    # if flags_obj.fp16_implementation == 'graph_rewrite':
    #   # Note: when flags/tmp/arion/scalability/models-master/official/vision/image_classification_obj.fp16_implementation == "graph_rewrite", dtype as
    #   # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
    #   # which will ensure tf.compat.v2.keras.mixed_precision and
    #   # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
    #   # up.
    #   optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
    #       optimizer)

    # TODO(hongkuny): Remove trivial model usage and move it to benchmark.
    if flags_obj.cnn_model == 'resnet101':
      model = tf.keras.applications.ResNet101(
        weights=None,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'vgg16':
      model = tf.keras.applications.VGG16(
        weights=None,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'inceptionv3':
      model = tf.keras.applications.InceptionV3(
        weights=None,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenet121':
      model = tf.keras.applications.DenseNet121(
        weights=None,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets121':
      model = DenseNet121(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets169':
      model = DenseNet169(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets201':
      model = DenseNet201(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets1':
      model = DenseNetS1(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets2':
      model = DenseNetS2(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets3':
      model = DenseNetS3(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets4':
      model = DenseNetS4(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets5':
      model = DenseNetS5(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets6':
      model = DenseNetS6(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets7':
      model = DenseNetS7(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets8':
      model = DenseNetS8(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets9':
      model = DenseNetS9(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets10':
      model = DenseNetS10(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenets11':
      model = DenseNetS11(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)      
    elif flags_obj.cnn_model == 'densenets12':
      model = DenseNetS12(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES) 
    elif flags_obj.cnn_model == 'densenets13':
      model = DenseNetS13(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES) 
    elif flags_obj.cnn_model == 'densenets14':
      model = DenseNetS14(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES) 
    elif flags_obj.cnn_model == 'densenets15':
      model = DenseNetS15(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)   
    elif flags_obj.cnn_model == 'densenets16':
      model = DenseNetS16(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)  
    elif flags_obj.cnn_model == 'densenets17':
      model = DenseNetS17(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)   
    elif flags_obj.cnn_model == 'densenets18':
      model = DenseNetS18(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)   
    elif flags_obj.cnn_model == 'densenets19':
      model = DenseNetS19(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)   
    elif flags_obj.cnn_model == 'densenets20':
      model = DenseNetS20(
        training=True,
        classes=imagenet_preprocessing.NUM_CLASSES)                                                           
    else:
      raise ValueError('Other Model Undeveloped')



    optimizer = tf.keras.optimizers.Adam(
      learning_rate=lr_schedule,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-08)

    train_input_iterator = tf.compat.v1.data.make_one_shot_iterator(train_input_dataset)
    train_input, train_target = train_input_iterator.get_next()

    # TODO(b/138957587): Remove when force_v2_in_keras_compile is on longer
    # a valid arg for this model. Also remove as a valid flag.
    # if flags_obj.force_v2_in_keras_compile is not None:
    #   model.compile(
    #       loss='sparse_categorical_crossentropy',
    #       optimizer=optimizer,
    #       metrics=(['sparse_categorical_accuracy']
    #                if flags_obj.report_accuracy_metrics else None),
    #       run_eagerly=flags_obj.run_eagerly,
    #       experimental_r/tmp/arion/scalability/models-master/official/vision/image_classificationun_tf_function=flags_obj.force_v2_in_keras_compile)
    # else:
    #   model.compile(
    #       loss='sparse_categorical_crossentropy',
    #       optimizer=optimizer,
    #       metrics=(['sparse_categorical_accuracy']
    #                if flags_obj.report_accuracy_metrics else None),
    #       run_eagerly=flags_obj.run_eagerly)

    steps_per_epoch = (
        imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
    train_epochs = flags_obj.train_epochs

    # callbacks = common.get_callbacks(steps_per_epoch,
    #                                  common.learning_rate_schedule)
    if flags_obj.enable_checkpoint_and_export:
      ckpt_full_path = os.path.join(flags_obj.model_dir, 'model.ckpt-{epoch:04d}')
      # callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
      #                                                     save_weights_only=True))

    # if mutliple epochs, ignore the train_steps flag.
    if train_epochs <= 1 and flags_obj.train_steps:
      steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)
      train_epochs = 1

    num_eval_steps = (
        imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)

    #validation_data = eval_input_dataset
    # if flags_obj.skip_eval:
    #   # Only build the training graph. This reduces memory usage introduced by
    #   # control flow ops in layers that have different implementations for
    #   # training and inference (e.g., batch norm).
    #   if flags_obj.set_learning_phase_to_train:
    #     # TODO(haoyuzhang): Understand slowdown of setting learning phase when
    #     # not using distribution strategy.
    #     tf.keras.backend.set_learning_phase(1)
    #   num_eval_steps = None
    #   validation_data = None

    train_output = model(train_input, training=True)
    scc_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    #loss = tf.compat.v1.keras.losses.sparse_categorical_crossentropy(train_target,
    # train_output)
    loss = scc_loss(train_target, train_output)
    var_list = variables.trainable_variables() + \
      ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
    grad = optimizer.get_gradients(loss, var_list)
    train_op = optimizer.apply_gradients(zip(grad, var_list))
    if flags_obj.is_arion:
      sess = arion.create_distributed_session()
    else:
      sess = tf.compat.v1.Session()
      sess.run(tf.compat.v1.global_variables_initializer()) 
    summary = TimeHistory(flags_obj.batch_size, steps_per_epoch)
    for epoch_id in range(train_epochs):
      summary.on_epoch_begin(epoch_id)
      for batch_id in range(steps_per_epoch):
        summary.on_batch_begin(batch_id)
        loss_v, _ = sess.run([loss, train_op])
        summary.on_batch_end(batch_id, loss_v)
      summary.on_epoch_end(epoch_id)
    summary.on_train_end()



    # if not strategy and flags_obj.explicit_gpu_placement:
    #   # TODO(b/135607227): Add device scope automatically in Keras training loop
    #   # when not using distribition strategy.
    #   no_dist_strat_device = tf.device('/device:GPU:0')
    #   no_dist_strat_device.__enter__()

    # history = model.fit(train_input_dataset,
    #                     epochs=train_epochs,
    #                     steps_per_epoch=steps_per_epoch,
    #                     callbacks=callbacks,
    #                     validation_steps=num_eval_steps,
    #                     validation_data=validation_data,
    #                     validation_freq=flags_obj.epochs_between_evals,
    #                     verbose=2)
    ###########TO DO: Save Ckpt###########
    # if flags_obj.enable_checkpoint_and_export:
    #   if dtype == tf.bfloat16:
    #     logging.warning("Keras model.save does not support bfloat16 dtype.")
    #   else:
    #     # Keras model.save assumes a float32 input designature.
    #     export_path = os.path.join(flags_obj.model_dir, 'saved_model')
    #     model.save(export_path, include_optimizer=False)

    ###########TO DO: Evaluation###########
    # eval_output = None
    # if not flags_obj.skip_eval:
    #   eval_output = model.evaluate(eval_input_dataset,
    #                                steps=num_eval_steps,
    #                                verbose=2)

    # if not strategy and flags_obj.explicit_gpu_placement:
    #   no_dist_strat_device.__exit__()

    #stats = common.build_stats(history, eval_output, callbacks)
  return

def define_imagenet_keras_flags():
  common.define_keras_flags()
  flags_core.set_defaults()
  flags.adopt_module_key_flags(common)


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  logdir = './logs'
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  if flags.FLAGS.is_arion:
    logname = 'log_strategy_{}_model_{}_node_{}_gpu_{}_patch_{}_chunk_{}'.format(flags.FLAGS.arion_strategy,
      flags.FLAGS.cnn_model, node_num, gpu_num, flags.FLAGS.arion_patch_tf, flags.FLAGS.chunk_size)
  else:
    logname = 'log_{}'.format(flags.FLAGS.cnn_model)
  logging.get_absl_handler().use_absl_log_file(logname, logdir)
  with logger.benchmark_context(flags.FLAGS):
    run(flags.FLAGS)
  #logging.info('Run stats:\n%s', stats)
  with open('end.o', 'w') as f:
    f.write('this test is done')
  exit()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_imagenet_keras_flags()
  app.run(main)

