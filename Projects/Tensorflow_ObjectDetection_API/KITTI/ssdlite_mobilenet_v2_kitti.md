
# 1、环境
source /home/jcq/anaconda3/bin/activate object_detection

cd /home/jcq/models-master/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:/home/jcq/models-master/research/:/home/jcq/models-master/research/slim:/home/jcq/models-master/research/slim/nets

/home/jcq/.conda/envs/object_detection/bin/python object_detection/builders/model_builder_test.py





# 2、数据集

## 2.1 训练钱数据准备

此时目录结构为

```
<20180823>
├── ssdlite_mobilenet_v2_coco_2018_05_09
│   └── saved_model
│       └── variables
├── mscoco
	└── *.record
├── kitti_label_map.pbtxt
└── faster_rcnn_resnet101_kitti.config
```

### 需要下载模型到该文件夹下


## 2.2、kitti 数据集

/media/jcq/Doc/DL_data/COCO_Data/KITTI



2、链接数据


  http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip.
  http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip

3、转成tfrecode




    python object_detection/dataset_tools/create_kitti_tf_record.py \
        --data_dir=/media/jcq/Doc/DL_data/KITTI \
        --output_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/KITTI/kitti.record \
        --label_map_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/KITTI/kitti_label_map.pbtxt


 ## train 的数据集没问题，但是eval 数据集不对啊，为0





# 3、使用 train.py 进行训练


/home/jcq/.conda/envs/object_detection/bin/python object_detection/legacy/train.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/KITTI/config/faster_rcnn_resnet101_kitti.config \
        --train_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/KITTI/models



/home/jcq/.conda/envs/object_detection/bin/python object_detection/legacy/train.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/KITTI/config/ssdlite_mobilenet_v2_coco.config \
        --train_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/KITTI/models



python object_detection/model_main.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/KITTI/config/ssdlite_mobilenet_v2_coco.config \
        --train_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/KITTI/models



## 3.1 、tensorboard 查看结果


tensorboard --logdir='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/model_0601'


## 3.2、断点续训功能，只修修改 config 文件中的 num_steps 即可继续训练




##3.3、eval 评估



/home/jcq/.conda/envs/object_detection/bin/python object_detection/legacy/eval.py \
        --logtostderr \
        --checkpoint_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/models \
        --eval_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/models \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/mscoco_label_map.pbtxt


        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/ssdlite_mobilenet_v2_coco.config \
        --train_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/models


##3.4、model_main 进行训练

python object_detection/model_main.py \
--logtostderr \
--pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/ssdlite_mobilenet_v2_coco.config \
--model_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/models1 \
--num_train_steps=50000 \
--num_eval_steps=2000



/home/jcq/.conda/envs/object_detection/bin/python object_detection/model_main.py \
--logtostderr \
--pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/config/ssd_mobilenet_v1_coco.config \
--model_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/models3 \
--num_train_steps=50000 \
--num_eval_steps=2000 

##(yes),终于能够跑了，成不成功另说


#3.5、更换模型


/home/jcq/.conda/envs/object_detection/bin/python object_detection/legacy/train.py \
        --logtostderr \
        --pipeline_config_path=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/config/ssd_mobilenet_v1_coco.config \
        --train_dir=/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/models2



# 4、导出训练模型做推理
在确定要导出的候选检查点之后，从`tensorflow/models/research`运行以下命令：

```python
cd tensorflow/models/research
# From tensorflow/models/research/

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/ssdlite_mobilenet_v2_coco.config'
TRAINED_CKPT_PREFIX='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/models/model.ckpt-206000' 
EXPORT_DIR='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/models/frozen_pb'
/home/jcq/.conda/envs/object_detection/bin/python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}


```

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/config/ssd_mobilenet_v1_coco.config'
TRAINED_CKPT_PREFIX='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/model_0601/model.ckpt-50000' 
EXPORT_DIR='/media/jcq/Soft/Tensorflow/Tensorflow_ObjectDetection_API/20200507/model_0601/frozen_pb'
/home/jcq/.conda/envs/object_detection/bin/python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}





# 5、 部署推理服务
  
  需要将 test.py 放在 object_detection 下面

/home/jcq/.conda/envs/object_detection/bin/python object_detection/test.py


## 完成,明显有效果

需要添加以下限制：

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))








# airunner 部分量化处理并部署



specify model and data folder location

export MODEL_DIR=/home/jcq/AIRunner/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/s32v234_sdk/libs/dnn/airunner/offline/quantization_tools/model/ssdlite_mb2
export DATA_DIR=/home/jcq/AIRunner/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/s32v234_sdk/libs/dnn/airunner/offline/quantization_tools/data

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

1. use transform_graph tool to remove pre/post processing part off mssd model

a) bazel build tensorflow/tools/graph_transforms:transform_graph

b) generate sub-model removing of pre/postprocessing parts

bazel-bin/tensorflow/tools/graph_transforms/transform_graph --in_graph=$MODEL_DIR/frozen_ssdlite_mb2.pb --out_graph=$MODEL_DIR/frozen_ssdlite_mb2_part.pb --inputs=Preprocessor/sub --outputs=concat,concat_1 --transforms='strip_unused_nodes(type=float, shape="1,300,300,3") remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true)'

c) apply batch normlization folding on sub model

bazel run --config=opt //tensorflow/contrib/lite/toco:toco -- --input_file=$MODEL_DIR/frozen_ssdlite_mb2_part.pb --output_file=$MODEL_DIR/frozen_ssdlite_mb2_part_bn.pb --input_format=TENSORFLOW_GRAPHDEF --output_format=TENSORFLOW_GRAPHDEF --input_shape=1,300,300,3 --input_array=Preprocessor/sub --output_arrays=concat,concat_1 --drop_control_dependency

d) quantized sub model  

python quantize_graph_mnet.py --input=$MODEL_DIR/frozen_ssdlite_mb2_part_bn.pb --output=$MODEL_DIR/frozen_ssdlite_mb2_part_bn_quant.pb --output_node_names=concat,concat_1 --mode=weights_sym_assign --print_nodes

2. verify model 

using following model verification tool need to follow below links to set up tensorflow object_detection first 
"protobuf compilation" and "add libraries to PYTHONPATH"

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md


2.a full float point model

python full_mssd_test.py --image=$DATA_DIR/image1.jpg --graph=$MODEL_DIR/frozen_ssdlite_mb2.pb --labels=$DATA_DIR/mscoco_label_map.pbtxt --outfile=$DATA_DIR/image1_annotated.png

2.b float point model removed of pre/post processing parts, connected with python implementation of pre/post processing 

python part_mssd_test.py --image=$DATA_DIR/image1.jpg --full_graph=$MODEL_DIR/frozen_ssdlite_mb2.pb --part_graph=$MODEL_DIR/frozen_ssdlite_mb2_part.pb --labels=$DATA_DIR/mscoco_label_map.pbtxt --outfile=$DATA_DIR/image1_annotated.png

2.c bn folded model removed of pre/post processing parts, connected with python implementation of pre/post processing

python part_mssd_test.py --image=$DATA_DIR/image1.jpg --full_graph=$MODEL_DIR/frozen_ssdlite_mb2.pb --part_graph=$MODEL_DIR/frozen_ssdlite_mb2_part_bn.pb --labels=$DATA_DIR/mscoco_label_map.pbtxt --outfile=$DATA_DIR/image1_annotated.png

2.d quantized bn folded model removed of pre/post processing parts, connected with python implementation of pre/post processing

python part_mssd_test.py --image=$DATA_DIR/image1.jpg --full_graph=$MODEL_DIR/frozen_ssdlite_mb2.pb --part_graph=$MODEL_DIR/frozen_ssdlite_mb2_part_bn_quant.pb --labels=$DATA_DIR/mscoco_label_map.pbtxt --outfile=$DATA_DIR/image1_annotated.png


3. final processing
python add_inputdequant.py --in_graph=$MODEL_DIR/frozen_ssdlite_mb2_part_bn_quant.pb --out_graph=$MODEL_DIR/frozen_ssdlite_mb2_part_bn_quant_final.pb

frozen_ssdlite_mb2_part_bn_quant_final.pb is the model to be used in apex application.




















