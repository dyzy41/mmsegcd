# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot
import cv2
import tqdm
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('file_list', help='image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-folder', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    model.dataset_meta['classes'] = ('Unknown', 'Bareland', 'Grass', 'Pavement', 'Road', 'Tree', 'Water', 'Cropland', 'Building')
    model.dataset_meta['palette'] = [[0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255], [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7]]
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    os.makedirs(args.out_folder, exist_ok=True)
    # test a single image
    file_list = open(args.file_list).readlines()
    for item in tqdm.tqdm(file_list):
        item = item.strip()
        img_path = item.split('  ')[0]
        img_name = os.path.basename(img_path).split('.')[0]+'.png'

        result = inference_model(model, img_path)
        # show the results
        show_result_pyplot(
            model,
            img_path,
            result,
            title=args.title,
            opacity=args.opacity,
            draw_gt=False,
            show=False if args.out_folder is not None else True,
            out_file=os.path.join(args.out_folder, img_name))    


if __name__ == '__main__':
    main()
