import os
import sys
import time


cmd1 = "python3 /tmp/arions/scalability/models-master/official/vision/image_classification/resnet_imagenet_main_arion.py --data_dir=/tmp/arions/scalability/train1 --train_epochs=10\
 --cnn_model={} --is_arion=True --arion_strategy={} --arion_patch_tf={}"


#for cnn_model in ['resnet101', 'densenet121', 'inceptionv3', 'vgg16']:
#    for arion_strategy in ['PS', 'PSLoadBalancing', 'PartitionedPS', 'AllReduce', 'Parallax']:
#for cnn_model in ['vgg16']:
#    for arion_strategy in ['SFB']:
        for arion_patch_tf in [True]:
            cmd = cmd1.format(cnn_model, arion_strategy, arion_patch_tf)
            print(cmd)
            os.system(cmd)
            while(not os.path.exists('/tmp/arions/scalability/models-master/official/vision/image_classification/end.o')):
                time.sleep(1)
            os.remove('/tmp/arions/scalability/models-master/official/vision/image_classification/end.o')