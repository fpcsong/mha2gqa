import torch
import os
import argparse

def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.system('cp {}/* {}'.format(args.init_model_path, args.output_path))

    state_dict = torch.load(os.path.join(args.model_path, 'pytorch_model.bin'))
    print('model loaded')
    new_dict = {}
    for k, param in state_dict.items():
        if 'student' in k:
            new_k = k.split('student_model.')[1]
            new_dict[new_k] = param
        # else:
        #     new_dict[k] = param
    print('new model generated')
    torch.save(new_dict, os.path.join(args.output_path, 'pytorch_model.bin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,\
                        default='', help='the folder contains model weights to convert')
    parser.add_argument('--init_model_path', type=str, required=True,\
                        default='', help='the folder contains init model weights')
    parser.add_argument('--output_path', type=str, required=True,\
                        default='', help='the folder contains target model weights')
    args = parser.parse_args()
    main(args)