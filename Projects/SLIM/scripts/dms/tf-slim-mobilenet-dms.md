
# run mobilenet_v1 on dms:  --karljiang-2019.12.23

source /home/jcq/anaconda3/bin/activate tensorflow_gpu

run  dms dataset_dir



## convert dataset to record

DATA_DIR=/media/jcq/Doc/DL_data/distracted_driver_detection/slim-dataset

/home/jcq/.conda/envs/tensorflow_gpu/bin/python download_and_convert_data.py \
    --dataset_name=dms \
    --dataset_dir="${DATA_DIR}"



CHECKPOINT_DIR=/media/jcq/Soft/Tensorflow/tensorflow-models/research/slim/tmp/checkpoints
cd ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
tar -xvf mobilenet_v1_1.0_224.tgz
rm mobilenet_v1_1.0_224.tgz
cd ../..


## set train steps 

DATASET_DIR=/media/jcq/Doc/DL_data/distracted_driver_detection/slim-dataset


### 选择不同的 models 进行训练

PRETRAINED_CHECKPOINT_DIR=/media/jcq/Soft/Tensorflow/tensorflow-models/research/slim/tmp/checkpoints/mobilenet_v1_1.0_224.ckpt
CHECKPOINT_PATH=/media/jcq/Soft/Tensorflow/tensorflow-models/research/slim/tmp/checkpoints/mobilenet_v1_1.0_224.ckpt

TRAIN_DIR=/media/jcq/Soft/Tensorflow/tensorflow-models/research/slim/tmp/dms-models/mobilenet_v1

1).
/home/jcq/.conda/envs/tensorflow_gpu/bin/python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=dms \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1 \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --checkpoint_exclude_scopes=MobilenetV1/Logits,MobilenetV1/AuxLogits \
  --trainable_scopes=MobilenetV1/Logits,MobilenetV1/AuxLogits \
  --max_number_of_steps=5000 \
  --batch_size=8 \
  --learning_rate=0.001 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004





# Evaluating performance of a model
# rename the validation files

/home/jcq/.conda/envs/tensorflow_gpu/bin/python eval_image_classifier.py \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=dms \
  --dataset_split_name=valid \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1

 ## 之前的错误属于数据集有问题，现在没有问题


INFO:tensorflow:Evaluation [1/7]
INFO:tensorflow:Evaluation [2/7]
INFO:tensorflow:Evaluation [3/7]
INFO:tensorflow:Evaluation [4/7]
INFO:tensorflow:Evaluation [5/7]
INFO:tensorflow:Evaluation [6/7]
INFO:tensorflow:Evaluation [7/7]

Train:1k
2019-12-31 08:45:11.611944: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.11857143]
2019-12-31 08:45:11.611944: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[0.53714287]


Train :10k

2020-01-02 11:02:28.157435: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.12714286]
2020-01-02 11:02:28.157452: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[0.5]



Train :50k

2020-01-06 09:48:15.028641: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.10857143]
2020-01-06 09:48:15.028649: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[0.48714286]



Train:100k

2020-04-30 09:44:40.621253: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.10857143]
2020-04-30 09:44:40.621265: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[0.48714286]

这二个已经没有任何区别了，训练几乎没有人任何区别，关键在于 Accuracy 和 Recall_5 的含义有什么区别呢？





# Fine-tune all the new layers for 50k steps.
/home/jcq/.conda/envs/tensorflow_gpu/bin/python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=dms \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${TRAIN_DIR} \
  --model_name=mobilenet_v1 \
  --max_number_of_steps=5000 \
  --batch_size=8 \
  --learning_rate=0.001 \
  --save_interval_secs=30 \
  --save_summaries_secs=30 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004





/home/jcq/.conda/envs/tensorflow_gpu/bin/python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=dms \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${TRAIN_DIR} \
  --model_name=mobilenet_v1 \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=1200 \
  --save_summaries_secs=1200 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004



log:
train 50k :
INFO:tensorflow:global step 49800: loss = 0.1387 (0.047 sec/step)
INFO:tensorflow:global step 49900: loss = 0.1548 (0.047 sec/step)
INFO:tensorflow:global step 50000: loss = 0.2176 (0.045 sec/step)


train 100k:
INFO:tensorflow:global step 99200: loss = 0.1798 (0.131 sec/step)
INFO:tensorflow:global step 99300: loss = 0.1580 (0.128 sec/step)
INFO:tensorflow:global step 99400: loss = 0.2088 (0.129 sec/step)
INFO:tensorflow:global step 99500: loss = 0.1494 (0.128 sec/step)
INFO:tensorflow:global step 99600: loss = 0.1850 (0.132 sec/step)
INFO:tensorflow:global step 99700: loss = 0.1247 (0.130 sec/step)
INFO:tensorflow:global step 99800: loss = 0.2076 (0.143 sec/step)
INFO:tensorflow:global step 99900: loss = 0.1596 (0.134 sec/step)
INFO:tensorflow:global step 100000: loss = 0.1235 (0.134 sec/step)

确实loss 下降了很多。



# Run evaluation.
/home/jcq/.conda/envs/tensorflow_gpu/bin/python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=dms \
  --dataset_split_name=valid \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1

log

INFO:tensorflow:Restoring parameters from /media/jcq/Soft/Tensorflow/tensorflow-models/research/slim/tmp/dms-models/mobilenet_v1/all/model.ckpt-50000
INFO:tensorflow:Evaluation [1/7]
INFO:tensorflow:Evaluation [2/7]
INFO:tensorflow:Evaluation [3/7]
INFO:tensorflow:Evaluation [4/7]
INFO:tensorflow:Evaluation [5/7]
INFO:tensorflow:Evaluation [6/7]
INFO:tensorflow:Evaluation [7/7]
2020-01-06 10:39:02.109075: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.99857146]
2020-01-06 10:39:02.109082: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[1]



INFO:tensorflow:Evaluation [1/7]
INFO:tensorflow:Evaluation [2/7]
INFO:tensorflow:Evaluation [3/7]
INFO:tensorflow:Evaluation [4/7]
INFO:tensorflow:Evaluation [5/7]
INFO:tensorflow:Evaluation [6/7]
INFO:tensorflow:Evaluation [7/7]
2020-04-30 14:27:09.765716: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[1]
2020-04-30 14:27:09.765729: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[1]

固然是准确率达到100

# Exporting the Inference Graph


/home/jcq/.conda/envs/tensorflow_gpu/bin/python export_inference_graph.py \
  --alsologtostderr \
  --dataset_dir=${DATASET_DIR} \
  --dataset_name=dms \
  --model_name=mobilenet_v1 \
  --image_size=224 \
  --output_file=/media/jcq/Soft/Tensorflow/tensorflow-models/research/slim/tmp/mb1_dms_inf_graph.pb


# Freezing the exported Graph



/home/jcq/.conda/envs/tensorflow_gpu/bin/python \
-u /home/jcq/.conda/envs/tensorflow_gpu/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py  \
  --input_graph=/media/jcq/Soft/Tensorflow/tensorflow-models/research/slim/tmp/mb1_dms_inf_graph.pb   \
  --input_checkpoint=/media/jcq/Soft/Tensorflow/tensorflow-models/research/slim/tmp/dms-models/mobilenet_v1/all/model.ckpt-5000 \
  --output_graph=/media/jcq/Soft/Tensorflow/tensorflow-models/research/slim/tmp/frozen_mb1_dms.pb   \
  --input_binary=True \
  --output_node_names=MobilenetV1/Predictions/Reshape_1








