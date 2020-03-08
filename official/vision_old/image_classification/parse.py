import os
import sys


for file in os.listdir('logs'):
    if file.endswith('INFO'):
        rowitem = []
        parse_name = file.split('.')[0].split('_')
        if len(parse_name) == 11:
            rowitem.extend([parse_name[2], parse_name[4], parse_name[6], parse_name[8], parse_name[10]])
        else:
            rowitem.extend(['tensorflow'], parse_name[1], None, None, None)
        with open('logs/{}'.format(file), 'r') as f:
            rls = f.readlines()
            rtime = rls[-3]
            if 'total time take' in rtime:
                rowitem.append(float(rtime.split('averaged examples_per_second: ')[1].split(',')[0][:-1]))
            else:
             continue
        print(rowitem)
