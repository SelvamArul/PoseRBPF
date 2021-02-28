import argparse
import torch
import json

from pathlib import Path

object_seq_info = {}
for i in range(1, 22):
    object_seq_info[f'obj_{i:06d}'] = []
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract_object_seq_info')
    parser.add_argument('results_path', help='path of the dataset')
    args = parser.parse_args()
    results_path = Path(args.results_path) / 'dataset=synpick' / 'results.pth.tar'

    results = torch.load(results_path)
    results_info = results['predictions']['maskrcnn_detections/detections'].infos

    for index, row in results_info.iterrows():
        label = row['label']
        scene_id = row['scene_id']
        view_id = row['view_id']
        d = {'index' : index,
            'label' : label,
            'scene_id' : scene_id,
            'view_id' : view_id}
        object_seq_info[label].append(d)

    for i in range(1, 22):
        obj = f'obj_{i:06d}'
        print ('obj ', obj)
        save_file = Path("/tmp") / f'{obj}.json'
        with open(save_file, 'w+') as jf:
            print ('# ', len(object_seq_info[obj]))
            d = {'entry': object_seq_info[obj]}
            json.dump(d, jf)






