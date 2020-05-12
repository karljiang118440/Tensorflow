#path set
export MODEL_DIR=/home/jcq/AIRunner/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/s32v234_sdk/libs/dnn/airunner/offline/quantization_tools/model/flowers
export DATA_DIR=/home/jcq/AIRunner/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/s32v234_sdk/libs/dnn/airunner/offline/quantization_tools/data



#对 mobilenet 进行的量化压缩




bazel run --config=opt //tensorflow/contrib/lite/toco:toco -- \
 --drop_control_depenency \
 --input_file=$MODEL_DIR/frozen_mb1_flowers.pb \
 --output_file=$MODEL_DIR/frozen_mb1_flowers_bn.pb \
 --input_format=TENSORFLOW_GRAPHDEF --output_format=TENSORFLOW_GRAPHDEF \
 --input_shape=1,224,224,3 --input_array=input --output_array=MobilenetV1/Predictions/Reshape_1


python mnet_imagelabelling_test-flowers.py \
--image=$DATA_DIR/rose.jpg  \
--graph=$MODEL_DIR/frozen_mb1_flowers.pb  \
--input_height=224 --input_width=224


roses 0.9999442
tulips 5.5327688e-05
daisy 3.178438e-07
dandelion 1.1159954e-07
sunflowers 4.4895227e-08




 
python quantize_graph_mnet.py \
 --input=$MODEL_DIR/frozen_mb1_flowers_bn.pb \
 --output_node_names=MobilenetV1/Predictions/Reshape_1 \
 --print_nodes --output=$MODEL_DIR/frozen_mb1_flowers_bn_qsym.pb \
 --mode=weights_sym –logtostderr
 


python mnet_imagelabelling_test-flowers.py \
 --image=$DATA_DIR/rose.jpg \
 --graph=$MODEL_DIR/frozen_mb1_flowers_bn.pb \
 --input_height=224 --input_width=224
 







python mnet_minmax_freeze.py \
 --in_graph=$MODEL_DIR/frozen_mb1_flowers_bn_qsym.pb \
 --out_graph=$MODEL_DIR/frozen_mb1_flowers_bn_qsym_final.pb \
 --out_graph_part=$MODEL_DIR/frozen_mb1_flowers_bn_qsym_final_part.pb \
 --output_layer=MobilenetV1/Predictions/Reshape_1 \
 --output_layer_part=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6





python mnet_imagelabelling_test-flowers.py \
 --image=$DATA_DIR/rose.jpg \
 --graph=$MODEL_DIR/frozen_mb1_flowers_bn_qsym.pb \
 --input_height=224 --input_width=224

python mnet_imagelabelling_test-flowers.py \
 --image=$DATA_DIR/rose.jpg \
 --graph=$MODEL_DIR/frozen_mb1_flowers_bn_qsym_final.pb \
 --input_height=224 --input_width=224
 
 
 python mnet_imagelabelling_test-flowers.py \
 --image=$DATA_DIR/rose.jpg \
 --graph=$MODEL_DIR/frozen_mb1_flowers_float_outputlayers_graph.pb \
 --input_height=224 --input_width=224








bazel run --config=opt //tensorflow/contrib/lite/toco:toco --\
 --drop_control_depenency\
 --input_file=$MODEL_DIR/frozen_mb1_flowers_bn.pb\
 --output_file=$MODEL_DIR/frozen_mb1_flowers_float_outputlayers_graph.pb\
 -input_format=TENSORFLOW_GRAPHDEF\
 --output_format=TENSORFLOW_GRAPHDEF\
 --input_shape=1,7,7,256\
 --input_array=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6\
 --output_arrays=MobilenetV1/Predictions/Reshape_1






## : compared the model accuracy in PC & s32v

python mnet_imagelabelling_test-flowers.py \
--image=$DATA_DIR/rose.jpg  \
--graph=$MODEL_DIR/frozen_mb1_flowers.pb  \
--labels=$MODEL_DIR/flowers-labels.txt  \
--input_height=224 --input_width=224


python mnet_imagelabelling_test-flowers.py \
--image=$DATA_DIR/rose.jpg  \
--graph=$MODEL_DIR/frozen_mb1_flowers.pb  \
--input_height=224 --input_width=224


python mnet_imagelabelling_test-flowers.py \
--image=$DATA_DIR/rose.jpg  \
--graph=$MODEL_DIR/frozen_mb1_flowers_bn.pb  \
--input_height=224 --input_width=224


python mnet_imagelabelling_test-flowers.py \
--image=$DATA_DIR/rose.jpg  \
--graph=$MODEL_DIR/frozen_mb1_flowers_bn_qsym.pb  \
--input_height=224 --input_width=224


python mnet_imagelabelling_test-flowers.py \
--image=$DATA_DIR/rose.jpg  \
--graph=$MODEL_DIR/frozen_mb1_flowers_bn_qsym_final.pb  \
--input_height=224 --input_width=224



roses 0.9999442
tulips 5.5327688e-05
daisy 3.178438e-07
dandelion 1.1159954e-07
sunflowers 4.4895227e-08

evb2中运行
               roses, 0.404598
               tulip, 0.148855
               daisy, 0.148849
           dandelion, 0.148849
          sunflowers, 0.148849

大概还是0.6的误差，很大了已经


