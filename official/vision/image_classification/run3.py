import os
import sys
import time


cmd1 = "python3 resnet_imagenet_main_arion.py --data_dir=/tmp/arion/scalability/train1 --train_epochs=10\
 --cnn_model={} --is_arion={}"
cmd2 = "python3 resnet_imagenet_main_arion.py --data_dir=/tmp/arion/scalability/train1 --train_epochs=10\
 --cnn_model={} --is_arion={} --arion_strategy={} --arion_patch_tf={}"


for cnn_model in ['inceptionv3', 'vgg16']:
    for is_arion in [True]:
        if is_arion is False:
            if cnn_model == 'vgg16':
                cmd = cmd1.format(cnn_model, is_arion)
                os.system(cmd)
                while(not os.path.exists('end.o')):
                    time.sleep(1)
                os.remove('end.o')
            else:
                continue
        else:
            #for arion_strategy in ['PS', 'PSLoadBalancing', 'PartitionedPS', 'AllReduce', 'Parallax']:
            for arion_strategy in ['PartitionedPS']:
                for arion_patch_tf in [True, False]:
                    cmd = cmd2.format(cnn_model, is_arion, arion_strategy, arion_patch_tf)
                    os.system(cmd)
                    while(not os.path.exists('end.o')):
                        time.sleep(1)
                    os.remove('end.o')
