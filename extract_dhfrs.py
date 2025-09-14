"""
A small script for computing and saving the DHFRs.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Sara Mandelli - sara.mandelli@polimi.it
Paolo Bestagini - paolo.bestagini@polimi.it
Stefano Tubaro - stefano.tubaro@polimi.it
"""


# --- Libraries import

import sys
import keras
import os
import numpy as np
import ntpath
import argparse
import sys
from isplutils.network import DnCNN
from isplutils.data import load_and_normalize
import pandas as pd
from typing import Tuple
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import tqdm


# --- Helpers function and classes
def extract_fingerprint(path: str, fe: keras.models.Model, use_histogram_equalization: bool):

    # Open the image
    img_float = load_and_normalize(path, use_histogram_equalization)

    # Predict the fingerprint
    res = fe.predict(img_float[np.newaxis, :, :, np.newaxis])
    res = np.squeeze(res)

    return res


def compute_fp(item: Tuple[pd.Index, pd.Series], root_dir: str, fe: keras.models.Model,
              save_path: str, use_histogram_equalization: bool) -> list:

    # Load true mask and sample
    idx, r = item
    img_path = r['filenames']

    # Extract the fingerprint
    noise = extract_fingerprint(os.path.join(root_dir, img_path), fe, use_histogram_equalization)

    # Save fingerprint
    img_name = ntpath.split(img_path)[1].split('.')[0]
    fp_path = os.path.join(save_path, 'fp_{}.npy'.format(img_name))
    np.save(fp_path, noise)

    # Update row
    r['fp_path'] = fp_path

    return [idx, r]


# ----------------------------------------- Main -----------------------------------------------------------------------
def main(args: argparse.Namespace):

    # --- Execution parameters --- #
    test_dir = args.test_dir
    fe_path = args.fe_path
    use_he = args.use_he
    results_dir = args.results_dir
    num = args.num
    offset = args.offset
    gpu = args.gpu
    debug = args.debug
    ops = args.ops

    # --- GPU configuration --- #
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # set the GPU device
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print('tf    version:', tf.__version__)
    print('keras version:', keras.__version__)

    configSess = tf.ConfigProto()
    # Allowing the GPU memory to grow avoiding preallocating all the GPU memory
    configSess.gpu_options.allow_growth = True
    set_session(tf.Session(config=configSess))

    # --- Load the test models --- #
    fe = DnCNN(model_path=fe_path)
    fe_tag = os.path.basename(fe_path).split('.')[0]

    # --- Load test dataset info DataFrame --- #
    test_df = pd.read_csv(os.path.join(test_dir, 'metadata.csv'), index_col=[0, 1])

    # --- Create main df for overall results --- #
    main_df = []

    # --- MAIN FOR LOOP --- #
    if ops is None:
        ops = test_df.index.unique(level=0)
    for op in ops:
        # We'll do operation by operation
        try:
            print('Starting with operation {}...'.format(op))
            op_df = test_df.loc[op]
            if debug:
                op_df = op_df.loc[np.unique(op_df.index.get_level_values(0)[:1])]
            else:
                if offset:
                    if num:
                        op_df = op_df.loc[np.unique(op_df.index.get_level_values(0)[offset:offset + num])]
                    else:
                        op_df = op_df.loc[np.unique(op_df.index.get_level_values(0)[offset:])]
                elif num:
                    op_df = op_df.loc[np.unique(op_df.index.get_level_values(0)[:num])]

            # Create save directory for the masks
            save_path = os.path.join(results_dir, fe_tag, op)
            masks_save_path = os.path.join(save_path, 'fingerprints')
            os.makedirs(masks_save_path, exist_ok=True)

            # Prepare test database
            result_df = op_df.copy()
            result_df['fp_path'] = np.nan

            # --- TEST LOOP --- #
            for idx, row in tqdm.tqdm(result_df.iterrows(), desc='Loading test images'):
                # Compute fingerprints
                _, result_row = compute_fp((idx, row), root_dir=test_dir, fe=fe,
                                                save_path=masks_save_path, use_histogram_equalization=use_he)
                result_df.loc[idx] = result_row

            # Save results per operation
            result_df = pd.concat({'{}'.format(op): result_df}, names=['Operation'])
            if debug:
                result_df.to_pickle(os.path.join(save_path, 'result_df_DEBUG.pkl'))
            else:
                if offset > 0:
                    if num > 0:
                        result_df_save_path = os.path.join(save_path,
                                                           'result_df_from_sample_{}_to_{}.pkl'.format(offset,
                                                                                                               num))
                    else:
                        result_df_save_path = os.path.join(save_path,
                                                           'result_df_from_sample_{}.pkl'.format(offset))
                elif num > 0:
                    result_df_save_path = os.path.join(save_path,
                                                       'result_df_from_sample_{}_to_{}.pkl'.format(offset, num))
                else:
                    result_df_save_path = os.path.join(save_path, 'result_df_complete.pkl')
                result_df.to_pickle(result_df_save_path)
            main_df.append(result_df.copy())
        except Exception as e:
            print('Something happened! Exception is {}'.format(e))

    # Save complete test results
    main_df = pd.concat(main_df)
    if args.ops is None:  # save the complete DataFrame only if analyzing all the operations together
        if debug:
            main_df.to_pickle(os.path.join(results_dir, fe_tag, 'all_ops_result_df_DEBUG.pkl'))
        else:
            if offset > 0:
                if num > 0:
                    main_df.to_pickle(os.path.join(results_dir,
                                                   fe_tag,
                                                   'all_ops_result_df_from_sample_{}_to_{}.pkl'.format(offset * len(ops),
                                                                                                       offset * len(ops) + num * len(ops))))
                else:
                    main_df.to_pickle(os.path.join(results_dir,
                                                   fe_tag,
                                                   'all_ops_result_df_from_sample_{}.pkl'.format(offset * len(ops))))
            elif num > 0:
                main_df.to_pickle(os.path.join(results_dir,
                                               fe_tag,
                                               'all_ops_result_df_from_sample_{}_to_{}.pkl'.format(0, num * len(ops))))
            else:
                main_df.to_pickle(os.path.join(results_dir, fe_tag, 'all_ops_result_df.pkl'))

    print('Results saved!')


if __name__ == '__main__':

    # --- Arguments parsing --- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3', help='GPU to use')
    parser.add_argument('--test_dir', type=str, default='data/test_samples',
                        help='Directory containing the single editing images')
    parser.add_argument('--fe_path', type=str,
                        default='./weights/asae.h5',
                        help='Fingerprint extractor considered')
    parser.add_argument('--use_he', action='store_true',
                        help='Use histogram equalization to normalize the test images')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--results_dir', type=str, default='test_results/',
                        help='Directory where to save the extracted fingerprints')
    parser.add_argument('--num', type=int, help='Number of samples per operation to analyze from the total', default=0)
    parser.add_argument('--offset', type=int, help='Offset for starting the samples analysis', default=0)
    parser.add_argument('--ops', type=str, nargs='+',
                        choices=['AdditiveWhiteGaussianNoise', 'AdditiveLaplacianNoise',
                                 'Affine', 'AverageBlur', 'MedianBlur', 'SpeckleNoise',], required=False,
                        help='Operation to consider for testing. If none, consider all of them')
    args = parser.parse_args()

    print('Executing the test for the whole pipeline...')
    main(args)
    print('Test finished! Bye!')
    sys.exit()
