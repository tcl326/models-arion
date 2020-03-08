import os
import sys
import time

cmd =  "python3 densenet_imagenet_main_arion.py --data_dir=/tmp/arion/scalability/train1 --train_epochs=10\
 --cnn_model={} --is_arion={} --arion_strategy={} --arion_patch_tf={} --chunk_size={}"


for cnn_model in ['densenets121']:
#for cnn_model in ['densenets121','densenets169','densenets201']:
    for arion_strategy in ['AllReduce', 'PS']:
        if arion_strategy == 'AllReduce':
            for chunk_size in [512]:
            #for chunk_size in [512,1024,2048,4096]:
                for arion_patch_tf in [True, False]:
                    cmd1 = cmd.format(cnn_model, True, arion_strategy, arion_patch_tf, chunk_size)
                    os.system(cmd1)
                    while(not os.path.exists('end.o')):
                        time.sleep(1)
                    os.remove('end.o')
        else:
            for arion_patch_tf in [True, False]:
                cmd1 = cmd.format(cnn_model, True, arion_strategy, arion_patch_tf, 512)
                os.system(cmd1)
                while(not os.path.exists('end.o')):
                    time.sleep(1)
                os.remove('end.o')
