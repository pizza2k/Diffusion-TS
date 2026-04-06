import os
import torch
import argparse
import numpy as np
import pandas as pd

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import cond_fn
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Sampling mode: infill, predict, cond, or unconditional when --sample 0.')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    parser.add_argument('--cond_label', type=int, default=0, help='Target label for classifier-guided conditional sampling.')
    parser.add_argument('--classifier_scale', type=float, default=1.0, help='Guidance scale for classifier-guided conditional sampling.')
    parser.add_argument('--classifier_milestone', type=int, default=None, help='Classifier checkpoint milestone. Defaults to --milestone.')
    parser.add_argument('--classifier_ckpt', type=str, default=None, help='Optional explicit path for classifier checkpoint.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    def log_array_stats(logger_obj, prefix, arr):
        if logger_obj is None:
            return
        if arr.size == 0:
            logger_obj.log_info(f'{prefix}: empty array')
            return
        logger_obj.log_info(
            '{} shape={} min={:.6f} max={:.6f} mean={:.6f}'.format(
                prefix,
                arr.shape,
                float(arr.min()),
                float(arr.max()),
                float(arr.mean()),
            )
        )

    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)
    train_dataset = dataloader_info['dataset']
    logger.log_info(
        'Train dataset info: name={} raw_len={} window={} var_num={} samples={} auto_norm={}'.format(
            train_dataset.name,
            train_dataset.len,
            train_dataset.window,
            train_dataset.var_num,
            len(train_dataset),
            train_dataset.auto_norm,
        )
    )

    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        log_array_stats(logger, 'Restore output (normalized space)', samples)
        samples = dataset.unnormalize(samples)
        log_array_stats(logger, 'Restore output (original scale)', samples)
        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}.npy'), samples)
    elif args.sample == 1 and args.mode in ['cond', 'conditional', 'class_cond']:
        if 'classifier' not in config:
            raise ValueError('Conditional sampling mode requires `classifier` config.')
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        column_names = dataloader_info['column_names']
        sample_batch_size = int(config['dataloader'].get('sample_size', config['dataloader']['batch_size']))

        classifier = instantiate_from_config(config['classifier']).cuda()
        classifier.eval()
        if args.classifier_ckpt is not None:
            classifier_ckpt_path = args.classifier_ckpt
        else:
            classifier_milestone = args.classifier_milestone if args.classifier_milestone is not None else args.milestone
            classifier_ckpt_path = str(trainer.results_folder / f'ckpt_classfier-{classifier_milestone}.pt')
        if not os.path.exists(classifier_ckpt_path):
            raise FileNotFoundError(f'Classifier checkpoint not found: {classifier_ckpt_path}')
        classifier_ckpt = torch.load(classifier_ckpt_path, map_location=model.device)
        classifier_state = classifier_ckpt.get('classifier', classifier_ckpt)
        classifier.load_state_dict(classifier_state)

        logger.log_info(
            'Start conditional sampling: mode={} label={} scale={} num={} size_every={} model_ckpt={} classifier_ckpt={}'.format(
                args.mode,
                args.cond_label,
                args.classifier_scale,
                len(dataset),
                sample_batch_size,
                args.milestone,
                classifier_ckpt_path,
            )
        )

        def build_model_kwargs(current_batch_size, _, __):
            return {
                'classifier': classifier,
                'y': torch.full((current_batch_size,), int(args.cond_label), device=model.device, dtype=torch.long),
                'classifier_scale': float(args.classifier_scale),
            }

        with torch.no_grad():
            samples = trainer.sample(
                num=len(dataset),
                size_every=sample_batch_size,
                shape=[dataset.window, dataset.var_num],
                model_kwargs_fn=build_model_kwargs,
                cond_fn=cond_fn,
            )
        log_array_stats(logger, 'Conditional sample output (normalized space)', samples)
        samples = dataset.unnormalize(samples)
        log_array_stats(logger, 'Conditional sample output (original scale)', samples)

        np.save(os.path.join(args.save_dir, f'ddpm_cond_fake_{args.name}.npy'), samples)
        _, _, feat_dim = samples.shape
        samples_2d = samples.reshape(-1, feat_dim)
        if len(column_names) != feat_dim:
            logger.log_info(
                'column_names length mismatch: {} vs {}, fallback to feature_i'.format(
                    len(column_names), feat_dim
                )
            )
            column_names = [f'feature_{i}' for i in range(feat_dim)]
        df = pd.DataFrame(samples_2d, columns=column_names)
        df.to_csv(os.path.join(args.save_dir, f'ddpm_cond_fake_{args.name}.csv'), index=False)
        logger.log_info('Saved conditional outputs to {}'.format(args.save_dir))
    elif args.sample == 1:
        raise ValueError('Unsupported conditional mode: {}. Use one of infill, predict, cond.'.format(args.mode))
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        column_names = dataloader_info['column_names']
        sample_batch_size = int(config['dataloader'].get('sample_size', config['dataloader']['batch_size']))
        logger.log_info(
            'Start unconditional sampling: num={} size_every={} milestone={}'.format(
                len(dataset), sample_batch_size, args.milestone
            )
        )

        samples = trainer.sample(
            num=len(dataset),
            size_every=sample_batch_size,
            shape=[dataset.window, dataset.var_num],
        )
        log_array_stats(logger, 'Sample output (normalized space)', samples)
        samples = dataset.unnormalize(samples)
        log_array_stats(logger, 'Sample output (original scale)', samples)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

        _, _, feat_dim = samples.shape
        samples_2d = samples.reshape(-1, feat_dim)
        if len(column_names) != feat_dim:
            logger.log_info(
                'column_names length mismatch: {} vs {}, fallback to feature_i'.format(
                    len(column_names), feat_dim
                )
            )
            column_names = [f'feature_{i}' for i in range(feat_dim)]
        df = pd.DataFrame(samples_2d, columns=column_names)
        df.to_csv(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.csv'), index=False)
        logger.log_info('Saved unconditional outputs to {}'.format(args.save_dir))

if __name__ == '__main__':
    main()
