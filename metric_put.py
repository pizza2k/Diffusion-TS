import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

        
import torch
import numpy as np
import pandas as pd

from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics
from Utils.context_fid import Context_FID
from Utils.cross_correlation import CrossCorrelLoss

from Utils.metric_utils import display_scores

def random_choice(size, num_select=100):
    select_idx = np.random.randint(low=0, high=size, size=(num_select,))
    return select_idx
    
def load_time_series_data(csv_path, seq_length=24, date_col='date'):
    df = pd.read_csv(csv_path)

    if date_col in df.columns:
        df = df.drop(columns=[date_col])

    data_2d = df.values  # 形状: (总行数, 特征数)
    
    # 4. 计算样本数
    total_rows = data_2d.shape[0]
    num_features = data_2d.shape[1]
    num_samples = total_rows // seq_length
    
    # 检查是否能完整分割
    if total_rows % seq_length != 0:
        print(f"警告: 总行数({total_rows})不能被seq_length({seq_length})整除")
        print(f"将丢弃最后 {total_rows % seq_length} 行")
        data_2d = data_2d[:num_samples * seq_length]
        num_samples = data_2d.shape[0] // seq_length
    
    # 5. 重塑为3D张量
    data_3d = data_2d.reshape(num_samples, seq_length, num_features)
    
    print(f"  - 样本数: {num_samples}")
    print(f"  - 序列长度: {seq_length}")
    print(f"  - 特征数: {num_features}")
    print(f"  - 输出形状: {data_3d.shape}")
    
    return data_3d

def put_metric(ori_csv,fake_csv,seq_len,name):
    ori_data = load_time_series_data(
        csv_path=ori_csv,
        seq_length=seq_len,  
        date_col='date'
    )
    
    fake_data = load_time_series_data(
        csv_path=fake_csv,
        seq_length=seq_len,
        date_col='date'
    )
    print(f'输出{name}相关的指标')
    iterations = 5
    
    ######### Context-CID
    
    context_fid_score = []

    for i in range(iterations):
        context_fid = Context_FID(ori_data[:], fake_data[:ori_data.shape[0]])
        context_fid_score.append(context_fid)
        print(f'Iter {i}: ', 'context-fid =', context_fid, '\n')
    print('Context-CID:')
    display_scores(context_fid_score)
    
    ####### correlational_score
    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(fake_data)
    
    correlational_score = []
    size = int(x_real.shape[0] / iterations)
    
    for i in range(iterations):
        real_idx = random_choice(x_real.shape[0], size)
        fake_idx = random_choice(x_fake.shape[0], size)
        corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
        loss = corr.compute(x_fake[fake_idx, :, :])
        correlational_score.append(loss.item())
        print(f'Iter {i}: ', 'cross-correlation =', loss.item(), '\n')
    print('cross-correlation:')
    display_scores(correlational_score)

    #####  discriminative_score
    discriminative_score = []

    for i in range(iterations):
        temp_disc, fake_acc, real_acc = discriminative_score_metrics(ori_data[:], fake_data[:ori_data.shape[0]])
        discriminative_score.append(temp_disc)
        print(f'Iter {i}: ', temp_disc, ',', fake_acc, ',', real_acc, '\n')
      
    print('discriminative_score:')
    display_scores(discriminative_score)
    print()

    ###### predictive_score
    predictive_score = []
    for i in range(iterations):
        temp_pred = predictive_score_metrics(ori_data, fake_data[:ori_data.shape[0]])
        predictive_score.append(temp_pred)
        print(i, ' epoch: ', temp_pred, '\n')
          
    print('predictive_score:')
    display_scores(predictive_score)
    print()
    
    
def main():
    put_metric('Data/datasets/tail_samples_tec_2014_192.csv','OUTPUT/tec_192/fake_tec_192.csv',704,'diffusion_tec_192')

if __name__ == '__main__':
    main()

