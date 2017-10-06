name: "CaffeNet"
input: 'data'
input_shape {
  dim: ${BATCH_SIZE}
  dim: ${SEQ_LEN*3}
  dim: ${CROP_SIZE}
  dim: ${CROP_SIZE}
}

input: 'imnet_label'
input_shape {
  dim: ${BATCH_SIZE}
  dim: ${SEQ_LEN}
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
    do_shuffle: false
  }
}

 % for seq_idx in range(SEQ_LEN):
# branch s${seq_idx} .................................................................................................

layer {
  name: "conv1_s${seq_idx}"
  type: "Convolution"
  bottom: "data_${seq_idx}"
  top: "conv1_s${seq_idx}"
  param {
    name: "conv1_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv1_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 2
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
layer {
  name: "relu1_s${seq_idx}"
  type: "ReLU"
  bottom: "conv1_s${seq_idx}"
  top: "conv1_s${seq_idx}"
}
layer {
  name: "pool1_s${seq_idx}"
  type: "Pooling"
  bottom: "conv1_s${seq_idx}"
  top: "pool1_s${seq_idx}"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm1_s${seq_idx}"
  type: "LRN"
  bottom: "pool1_s${seq_idx}"
  top: "norm1_s${seq_idx}"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv2_s${seq_idx}"
  type: "Convolution"
  bottom: "norm1_s${seq_idx}"
  top: "conv2_s${seq_idx}"
  param {
    name: "conv2_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv2_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu2_s${seq_idx}"
  type: "ReLU"
  bottom: "conv2_s${seq_idx}"
  top: "conv2_s${seq_idx}"
}
layer {
  name: "pool2_s${seq_idx}"
  type: "Pooling"
  bottom: "conv2_s${seq_idx}"
  top: "pool2_s${seq_idx}"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm2_s${seq_idx}"
  type: "LRN"
  bottom: "pool2_s${seq_idx}"
  top: "norm2_s${seq_idx}"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv3_s${seq_idx}"
  type: "Convolution"
  bottom: "norm2_s${seq_idx}"
  top: "conv3_s${seq_idx}"
  param {
    name: "conv3_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv3_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
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
layer {
  name: "relu3_s${seq_idx}"
  type: "ReLU"
  bottom: "conv3_s${seq_idx}"
  top: "conv3_s${seq_idx}"
}
layer {
  name: "conv4_s${seq_idx}"
  type: "Convolution"
  bottom: "conv3_s${seq_idx}"
  top: "conv4_s${seq_idx}"
  param {
    name: "conv4_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv4_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu4_s${seq_idx}"
  type: "ReLU"
  bottom: "conv4_s${seq_idx}"
  top: "conv4_s${seq_idx}"
}
layer {
  name: "conv5_s${seq_idx}"
  type: "Convolution"
  bottom: "conv4_s${seq_idx}"
  top: "conv5_s${seq_idx}"
  param {
    name: "conv5_weigths"
    lr_mult: 1
    decay_mult: 1
  }
  param {
    name: "conv5_bias"
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu5_s${seq_idx}"
  type: "ReLU"
  bottom: "conv5_s${seq_idx}"
  top: "conv5_s${seq_idx}"
}
layer {
  name: "pool5_s${seq_idx}"
  type: "Pooling"
  bottom: "conv5_s${seq_idx}"
  top: "pool5_s${seq_idx}"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc6_s${seq_idx}"
  type: "InnerProduct"
  bottom: "pool5_s${seq_idx}"
  top: "fc6_s${seq_idx}"
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
  name: "relu6_s${seq_idx}"
  type: "ReLU"
  bottom: "fc6_s${seq_idx}"
  top: "fc6_s${seq_idx}"
}
layer {
  name: "drop6_s${seq_idx}"
  type: "Dropout"
  bottom: "fc6_s${seq_idx}"
  top: "fc6_s${seq_idx}"
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
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6_concat"
  top: "fc7"
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
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "fc7"
  top: "fc7"
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
  top: "prob"
  sinkhorn_norm_param{
     is_row_step: false
     num_rows: ${SEQ_LEN}
     num_cols: ${SEQ_LEN}
     lambda: 0.01
  }
}


