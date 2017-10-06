# MALE
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ Male --seq_len 2 4 6 8 2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_male.log

# White
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ White --seq_len 8 6 4 2 2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_white.log

# Young
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ Young --seq_len 2 4 6 8 2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_young.log

# Smiling
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ Smiling --seq_len 2 4 6 8 2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_smiling.log

# Chubby
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ Chubby --seq_len 2 4 6 8  2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_chubby.log

# VisibleForehead
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ VisibleForehead --seq_len 2 4 6 8  2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_visibleForehead.log

# BushyEyebrows
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ BushyEyebrows --seq_len 2 4 6 8  2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_bushyEyebrows.log

# NarrowEyes
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ NarrowEyes --seq_len 2 4 6 8 2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_narrowEyes.log

# PointyNose
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ PointyNose --seq_len 2 4 6 8 2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_pointyNose.log

# BigLips
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ BigLips --seq_len 2 4 6 8 2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_biglips.log

# RoundFace
python ./lib/attributes/relatt_train_test.py  pub_fig /home/rfsc/Datasets/relative_attributes/pubfig/ ./lib/attributes/ ./output/attributes/sinkhorn/pubfig/ RoundFace --seq_len 2 4 6 8 2>&1 | tee -a ./output/attributes/sinkhorn/pubfig/exp_roundFace.log
