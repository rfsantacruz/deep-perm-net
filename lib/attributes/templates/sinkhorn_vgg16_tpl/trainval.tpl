name: "CaffeNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "imnet_label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: false
    crop_size: ${CROP_SIZE}
    % for seq_idx in range(SEQ_LEN):
    	mean_value: 104
    	mean_value: 117
    	mean_value: 123
    % endfor    
  }
  data_param {
    source: "${TRAIN_DB_PATH}"
    batch_size: ${BATCH_SIZE}
    backend: LMDB
  }
}

layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "imnet_label"
  include {
    phase: TEST
  }
   transform_param {
    mirror: false
    crop_size: ${CROP_SIZE}
    % for seq_idx in range(SEQ_LEN):
    	mean_value: 104
    	mean_value: 117
    	mean_value: 123
    % endfor    
  }
  data_param {
    source: "${TEST_DB_PATH}"
    batch_size: ${BATCH_SIZE}
    backend: LMDB
  }
}

layer {
  name: 'perm_gen'
  type: 'PermMatrixSequence'
  bottom: 'data'
  bottom: 'imnet_label'
  % for seq_idx in range(SEQ_LEN):
    	top: 'data_${seq_idx}'
  % endfor
  top: 'labels'
  perm_matrix_sequence_param {
    seq_len: ${SEQ_LEN}
  }
}

 % for seq_idx in range(SEQ_LEN):
# branch s${seq_idx} .................................................................................................
layer {
  type: "Convolution"
  bottom: "data_${seq_idx}"
  top: "conv1_1_s${seq_idx}"
  name: "conv1_1_s${seq_idx}"
  param {
    name: "conv1_1_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv1_1_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 3
    pad: 1
  }
}

layer {
  bottom: "conv1_1_s${seq_idx}"
  top: "conv1_1_s${seq_idx}"
  name: "relu1_1_s${seq_idx}"
  type: "ReLU"
}

layer {
  bottom: "conv1_1_s${seq_idx}"
  top: "conv1_2_s${seq_idx}"
  name: "conv1_2_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv1_2_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv1_2_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 3
    pad: 1
  }
}
layer {
  bottom: "conv1_2_s${seq_idx}"
  top: "conv1_2_s${seq_idx}"
  name: "relu1_2_s${seq_idx}"
  type: "ReLU"
}
layer {
  bottom: "conv1_2_s${seq_idx}"
  top: "pool1_s${seq_idx}"
  name: "pool1_s${seq_idx}"
  type: "Pooling"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  bottom: "pool1_s${seq_idx}"
  top: "conv2_1_s${seq_idx}"
  name: "conv2_1_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv2_1_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv2_1_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
  }
}

layer {
  bottom: "conv2_1_s${seq_idx}"
  top: "conv2_1_s${seq_idx}"
  name: "relu2_1_s${seq_idx}"
  type: "ReLU"
}

layer {
  bottom: "conv2_1_s${seq_idx}"
  top: "conv2_2_s${seq_idx}"
  name: "conv2_2_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv2_2_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv2_2_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
  }
}
layer {
  bottom: "conv2_2_s${seq_idx}"
  top: "conv2_2_s${seq_idx}"
  name: "relu2_2_s${seq_idx}"
  type: "ReLU"
}
layer {
  bottom: "conv2_2_s${seq_idx}"
  top: "pool2_s${seq_idx}"
  name: "pool2_s${seq_idx}"
  type: "Pooling"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  bottom: "pool2_s${seq_idx}"
  top: "conv3_1_s${seq_idx}"
  name: "conv3_1_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv3_1_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv3_1_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}

layer {
  bottom: "conv3_1_s${seq_idx}"
  top: "conv3_1_s${seq_idx}"
  name: "relu3_1_s${seq_idx}"
  type: "ReLU"
}

layer {
  bottom: "conv3_1_s${seq_idx}"
  top: "conv3_2_s${seq_idx}"
  name: "conv3_2_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv3_2_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv3_2_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}

layer {
  bottom: "conv3_2_s${seq_idx}"
  top: "conv3_2_s${seq_idx}"
  name: "relu3_2_s${seq_idx}"
  type: "ReLU"
}

layer {
  bottom: "conv3_2_s${seq_idx}"
  top: "conv3_3_s${seq_idx}"
  name: "conv3_3_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv3_3_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv3_3_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}
layer {
  bottom: "conv3_3_s${seq_idx}"
  top: "conv3_3_s${seq_idx}"
  name: "relu3_3_s${seq_idx}"
  type: "ReLU"
}
layer {
  bottom: "conv3_3_s${seq_idx}"
  top: "pool3_s${seq_idx}"
  name: "pool3_s${seq_idx}"
  type: "Pooling"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  bottom: "pool3_s${seq_idx}"
  top: "conv4_1_s${seq_idx}"
  name: "conv4_1_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv4_1_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv4_1_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}

layer {
  bottom: "conv4_1_s${seq_idx}"
  top: "conv4_1_s${seq_idx}"
  name: "relu4_1_s${seq_idx}"
  type: "ReLU"
}
layer {
  bottom: "conv4_1_s${seq_idx}"
  top: "conv4_2_s${seq_idx}"
  name: "conv4_2_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv4_2_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv4_2_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  bottom: "conv4_2_s${seq_idx}"
  top: "conv4_2_s${seq_idx}"
  name: "relu4_2_s${seq_idx}"
  type: "ReLU"
}
layer {
  bottom: "conv4_2_s${seq_idx}"
  top: "conv4_3_s${seq_idx}"
  name: "conv4_3_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv4_3_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv4_3_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  bottom: "conv4_3_s${seq_idx}"
  top: "conv4_3_s${seq_idx}"
  name: "relu4_3_s${seq_idx}"
  type: "ReLU"
}
layer {
  bottom: "conv4_3_s${seq_idx}"
  top: "pool4_s${seq_idx}"
  name: "pool4_s${seq_idx}"
  type: "Pooling"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  bottom: "pool4_s${seq_idx}"
  top: "conv5_1_s${seq_idx}"
  name: "conv5_1_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv5_1_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv5_1_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  bottom: "conv5_1_s${seq_idx}"
  top: "conv5_1_s${seq_idx}"
  name: "relu5_1_s${seq_idx}"
  type: "ReLU"
}
layer {
  bottom: "conv5_1_s${seq_idx}"
  top: "conv5_2_s${seq_idx}"
  name: "conv5_2_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv5_2_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv5_2_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}

layer {
  bottom: "conv5_2_s${seq_idx}"
  top: "conv5_2_s${seq_idx}"
  name: "relu5_2_s${seq_idx}"
  type: "ReLU"
}

layer {
  bottom: "conv5_2_s${seq_idx}"
  top: "conv5_3_s${seq_idx}"
  name: "conv5_3_s${seq_idx}"
  type: "Convolution"
  param {
    name: "conv5_3_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv5_3_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  bottom: "conv5_3_s${seq_idx}"
  top: "conv5_3_s${seq_idx}"
  name: "relu5_3_s${seq_idx}"
  type: "ReLU"
}
layer {
  bottom: "conv5_3_s${seq_idx}"
  top: "pool5_s${seq_idx}"
  name: "pool5_s${seq_idx}"
  type: "Pooling"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  bottom: "pool5_s${seq_idx}"
  top: "fc6_s${seq_idx}"
  name: "fc6_s${seq_idx}"
  type: "InnerProduct"
  param {
    name: "fc6_weigths"
    lr_mult: 10
    decay_mult: 1
  }
  param {
    name: "fc6_bias"
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: 512
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  bottom: "fc6_s${seq_idx}"
  top: "fc6_s${seq_idx}"
  name: "relu6_s${seq_idx}"
  type: "ReLU"
}
layer {
  bottom: "fc6_s${seq_idx}"
  top: "fc6_s${seq_idx}"
  name: "drop6_s${seq_idx}"
  type: "Dropout"
  dropout_param {
    dropout_ratio: 0.5
  }
}

# ...........................................................................................

% endfor

layer {
  name: "concat"
% for seq_idx in range(SEQ_LEN):
  bottom: "fc6_s${seq_idx}"
% endfor
  top: "fc6_concat"
% if FUSION:
  type: "SequenceFusion"
% else:
  type: "Concat"
  concat_param {
    axis: 1
  }
% endif
}

layer {
  bottom: "fc6_concat"
  top: "fc7"
  name: "fc7"
  type: "InnerProduct"
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}

layer {
  bottom: "fc7"
  top: "fc7"
  name: "relu7"
  type: "ReLU"
}
layer {
  bottom: "fc7"
  top: "fc7"
  name: "drop7"
  type: "Dropout"
  dropout_param {
    dropout_ratio: 0.5
  }
}

layer {
  name: "fc8"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8"
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: ${pow(SEQ_LEN,2)}
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}

layer{
  name: "norm"
  type: "Softmax"
  bottom: "fc8"
  top: "fc8_s"
}

layer{
  name: "norm_row1"
  type: "SinkhornNorm"
  bottom: "fc8_s"
  top: "fc8_r1"
  sinkhorn_norm_param{
     is_row_step: true
     num_rows: ${SEQ_LEN}
     num_cols: ${SEQ_LEN}
     lambda: 0.01
  }
}

layer{
  name: "norm_col1"
  type: "SinkhornNorm"
  bottom: "fc8_r1"
  top: "fc8_c1"
  sinkhorn_norm_param{
     is_row_step: false
     num_rows: ${SEQ_LEN}
     num_cols: ${SEQ_LEN}
     lambda: 0.01
  }
}



layer {
  name: "loss_train"
  type: "MultilabelMultinomialLogisticLoss"
  bottom: "fc8_c1"
  bottom: "labels"
  top: "loss_train"
  loss_weight: 1 
  include {
    phase: TRAIN
  } 
}


layer {
  name: "loss_val"
  type: "MultilabelMultinomialLogisticLoss"
  bottom: "fc8_c1"
  bottom: "labels"
  top: "loss_val"
  loss_weight: 1
  include {
    phase: TEST
  }
}
