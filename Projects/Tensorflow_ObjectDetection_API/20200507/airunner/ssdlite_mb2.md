
#对 frozen_ssdlite_mb2 进行的量化压缩




tested on tensorflow d836210e7d7c8bf54676fd4154f40920310cdb27 and Python 2.7.12


specify model and data folder location

export MODEL_DIR=/home/jcq/AIRunner/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/s32v234_sdk/libs/dnn/airunner/offline/quantization_tools/model/ssdlite_mb2_person
export DATA_DIR=/home/jcq/AIRunner/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/SW32V23-VSDK-AIRUNNER-CODE_DROP-1.2.0-2/s32v234_sdk/libs/dnn/airunner/offline/quantization_tools/data



https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# 1. use transform_graph tool to remove pre/post processing part off mssd model

a) bazel build tensorflow/tools/graph_transforms:transform_graph

b) generate sub-model removing of pre/postprocessing parts

bazel-bin/tensorflow/tools/graph_transforms/transform_graph\
 --in_graph=$MODEL_DIR/frozen_ssdlite_mb2.pb\
 --out_graph=$MODEL_DIR/frozen_ssdlite_mb2_part.pb\
 --inputs=Preprocessor/sub\
 --outputs=concat,concat_1\
 --transforms='strip_unused_nodes(type=float, shape="1,300,300,3") remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true)'






c) apply batch normlization folding on sub model

bazel run --config=opt //tensorflow/contrib/lite/toco:toco\
 -- --input_file=$MODEL_DIR/frozen_ssdlite_mb2_part.pb\
 --output_file=$MODEL_DIR/frozen_ssdlite_mb2_part_bn.pb\
 --input_format=TENSORFLOW_GRAPHDEF\
 --output_format=TENSORFLOW_GRAPHDEF\
 --input_shape=1,300,300,3\
 --input_array=Preprocessor/sub\
 --output_arrays=concat,concat_1\
 --drop_control_dependency

d) quantized sub model  

python quantize_graph_mnet.py\
 --input=$MODEL_DIR/frozen_ssdlite_mb2_part_bn.pb\
 --output=$MODEL_DIR/frozen_ssdlite_mb2_part_bn_quant.pb\
 --output_node_names=concat,concat_1 --mode=weights_sym_assign --print_nodes

2. verify model 

using following model verification tool need to follow below links to set up tensorflow object_detection first 
"protobuf compilation" and "add libraries to PYTHONPATH"

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md


2.a full float point model

python full_mssd_test.py\
 --image=$DATA_DIR/image1.jpg\
 --graph=$MODEL_DIR/frozen_ssdlite_mb2.pb\
 --labels=$DATA_DIR/mscoco_label_map.pbtxt\
 --outfile=$DATA_DIR/image1_annotated.png

2.b float point model removed of pre/post processing parts, connected with python implementation of pre/post processing 

python part_mssd_test.py\
 --image=$DATA_DIR/image1.jpg\
 --full_graph=$MODEL_DIR/frozen_ssdlite_mb2.pb\
 --part_graph=$MODEL_DIR/frozen_ssdlite_mb2_part.pb\
 --labels=$DATA_DIR/mscoco_label_map.pbtxt\
 --outfile=$DATA_DIR/image1_annotated.png

2.c bn folded model removed of pre/post processing parts, connected with python implementation of pre/post processing

python part_mssd_test.py\
 --image=$DATA_DIR/image1.jpg --full_graph=$MODEL_DIR/frozen_ssdlite_mb2.pb --part_graph=$MODEL_DIR/frozen_ssdlite_mb2_part_bn.pb --labels=$DATA_DIR/mscoco_label_map.pbtxt --outfile=$DATA_DIR/image1_annotated.png

2.d quantized bn folded model removed of pre/post processing parts, connected with python implementation of pre/post processing

python part_mssd_test.py --image=$DATA_DIR/image1.jpg --full_graph=$MODEL_DIR/frozen_ssdlite_mb2.pb --part_graph=$MODEL_DIR/frozen_ssdlite_mb2_part_bn_quant.pb --labels=$DATA_DIR/mscoco_label_map.pbtxt --outfile=$DATA_DIR/image1_annotated.png


3. final processing
python add_inputdequant.py\
 --in_graph=$MODEL_DIR/frozen_ssdlite_mb2_part_bn_quant.pb\
 --out_graph=$MODEL_DIR/frozen_ssdlite_mb2_part_bn_quant_final.pb

frozen_ssdlite_mb2_part_bn_quant_final.pb is the model to be used in apex application.

