#importing libraries
import csv
import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from urllib.request import urlopen

if __name__ == "__main__":
    files = os.listdir('../DATA/YT8M/.')
    train_files = [ x for x in files if x.startswith('train')]
    valid_files = [ x for x in files if x.startswith('valid')]
    test_files = [ x for x in files if x.startswith('test')]

    print(len(train_files), 'training files')
    print(len(valid_files), 'validation files')
    print(len(test_files), 'test files')

    final_pseudoids = []
    for sets, name_sets in zip([train_files, valid_files, test_files], ['train', 'valid', 'test']):
        pseudo_ids = []
        for f in sets:
            # print(f)
            for example in tf.python.python_io.tf_record_iterator("YT8M/"+f):
                tf_example = tf.train.Example.FromString(example)
                pseudo_ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))


        print('{} {} video pseudo ids'.format(len(pseudo_ids), name_sets))
        set_pseudoids = set(pseudo_ids)
        list_pseudoids = list(set_pseudoids)
        print('{} {} unique video pseudo ids'.format(len(list_pseudoids), name_sets))

        final_pseudoids.extend(list_pseudoids)

    print('{} total video pseudo ids'.format(len(final_pseudoids)))
    set_final_pseudoids = set(final_pseudoids)
    final_pseudoids = list(set_final_pseudoids)
    print('{} total unique video pseudo ids'.format(len(final_pseudoids)))
    df = pd.DataFrame(final_pseudoids, columns =['pseudoId'])
    df.to_csv('../DATA/videospseudoID8M.csv', encoding='utf-8', index=False)
