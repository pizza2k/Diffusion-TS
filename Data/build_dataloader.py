import torch
from Utils.io_utils import instantiate_from_config


def build_dataloader(config, args=None):
    batch_size = config['dataloader']['batch_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['train_dataset']['params']['output_dir'] = args.save_dir
    dataset = instantiate_from_config(config['dataloader']['train_dataset'])
    column_names = getattr(dataset, 'column_names', None)
    if column_names is None:
        column_names = [f'feature_{i}' for i in range(dataset.var_num)]
        dataset.column_names = column_names

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': dataset,
        'column_names': column_names,
    }

    return dataload_info

def build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['sample_size']
    config['dataloader']['test_dataset']['params']['output_dir'] = args.save_dir
    if args.mode == 'infill':
        config['dataloader']['test_dataset']['params']['missing_ratio'] = args.missing_ratio
    elif args.mode == 'predict':
        config['dataloader']['test_dataset']['params']['predict_length'] = args.pred_len
    test_dataset = instantiate_from_config(config['dataloader']['test_dataset'])
    column_names = getattr(test_dataset, 'column_names', None)
    if column_names is None:
        column_names = [f'feature_{i}' for i in range(test_dataset.var_num)]
        test_dataset.column_names = column_names

    dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=False)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': test_dataset,
        'column_names': column_names,
    }

    return dataload_info


if __name__ == '__main__':
    pass

