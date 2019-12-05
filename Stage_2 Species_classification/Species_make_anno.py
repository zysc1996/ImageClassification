import pandas as pd
import os
from PIL import Image

ROOTS = '../Dataset/'
PHASE = ['train', 'val']
SPECIES = ['rabbits', 'rats', 'chickens']  # [0,1,2]

DATA_info = {'train': {'path': [], 'species': []},
             'val': {'path': [], 'species': []}
             }
for p in PHASE:
    for s in SPECIES:
        DATA_DIR = ROOTS + p + '\\' + s
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
                elif s == 'rats':
                    DATA_info[p]['species'].append(1)
                else:
                    DATA_info[p]['species'].append(2)

    ANNOTATION = pd.DataFrame(DATA_info[p])
    ANNOTATION.to_csv('Species_%s_annotation.csv' % p)
    print('Species_%s_annotation file is saved.' % p)
