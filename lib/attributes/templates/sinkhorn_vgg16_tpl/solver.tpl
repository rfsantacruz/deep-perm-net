% if TRAIN_VAL:
net: "${TRAINVAL_FILE}"
test_iter: ${TEST_ITERS}
test_interval: ${TEST_INT}
% else:
train_net: "${TRAINVAL_FILE}"
% endif
base_lr: ${LR}
lr_policy: "step"
gamma: 0.1
stepsize: ${LR_STEP}
display: 20
max_iter: ${MAX_ITERS}
momentum: 0.9
weight_decay: 0.0005
average_loss: 100
snapshot: ${SNAPSHOT_ITERS}
snapshot_prefix: "${SNAPSHOT_DIR}"
