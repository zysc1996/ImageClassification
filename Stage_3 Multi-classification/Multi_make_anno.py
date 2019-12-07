import pandas as pd
import os
from PIL import Image

# 多标签，这里分别用0，1对classes，用0，1，2对species做标签
ROOTS = '../Dataset/'
PHASE = ['train', 'val']
CLASSES = ['Mammals', 'Birds']  # [0,1]
SPECIES = ['rabbits', 'rats', 'chickens']  # [0,1,2]

DATA_info = {'train': {'path': [], 'classes':[],'species': []},
             'val': {'path': [], 'classes':[],'species': []}
             }
for p in PHASE:
    for s in SPECIES:
        DATA_DIR = ROOTS + p + '/' + s
        DATA_NAME = os.listdir(DATA_DIR)

        for item in DATA_NAME:
            try:
                img = Image.open(os.path.join(DATA_DIR, item))
            except OSError:
                pass
            else:
                DATA_info[p]['path'].append(os.path.join(DATA_DIR, item))

                if s == 'rabbits':
                    DATA_info[p]['species'].append(0)
                    DATA_info[p]['classes'].append(0)
                elif s == 'rats':
                    DATA_info[p]['species'].append(1)
                    DATA_info[p]['classes'].append(0)
                else:
                    DATA_info[p]['species'].append(2)
                    DATA_info[p]['classes'].append(1)

    ANNOTATION = pd.DataFrame(DATA_info[p])
    # 生成一个由两个数字组成的annotation的csv文件
    ANNOTATION.to_csv('Multi_%s_annotation.csv' % p)
    print('Multi_%s_annotation file is saved.' % p)