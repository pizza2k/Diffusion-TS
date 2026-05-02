import pandas as pd
from Utils.metric_utils import visualization

print('start')

'''
Data/datasets/tail_samples_ETTh1_0_96.csv
Data/datasets/tail_samples_ETTh1_0_96.csv
Data/datasets/tail_samples_tec_2014_96.csv
Data/datasets/tail_samples_tec_2014_192.csv

OUTPUT/etth1_96/fake_etth1_96.csv
OUTPUT/etth1_192/fake_etth1_192.csv

'''

ori = [
    # 'Data/datasets/tail_samples_ETTh1_0_96.csv','Data/datasets/tail_samples_ETTh1_0_192.csv','Data/datasets/tail_samples_ETTh1_0_336.csv',  'Data/datasets/tail_samples_ETTh2_0_96.csv','Data/datasets/tail_samples_ETTh2_0_192.csv','Data/datasets/tail_samples_ETTh2_0_336.csv', 'Data/datasets/tail_samples_tec_2014_96.csv','Data/datasets/tail_samples_tec_2014_192.csv','Data/datasets/tail_samples_tec_2014_336.csv',
    'Data/datasets/tail_samples_weather_0_96.csv','Data/datasets/tail_samples_weather_0_192.csv','Data/datasets/tail_samples_weather_0_336.csv'
]

gen = [
    # 'OUTPUT/etth1_96/fake_etth1_96.csv','OUTPUT/etth1_192/fake_etth1_192.csv','OUTPUT/etth1_336/fake_etth1_336.csv',
    # 'OUTPUT/etth2_96/fake_etth2_96.csv','OUTPUT/etth2_192/fake_etth2_192.csv','OUTPUT/etth2_336/fake_etth2_336.csv',
    # 'OUTPUT/tec_96/fake_tec_96.csv','OUTPUT/tec_192/fake_tec_192.csv','OUTPUT/tec_336/fake_tec_336.csv'
    'OUTPUT/weather_96/fake_weather_96.csv','OUTPUT/weather_192/fake_weather_192csv','OUTPUT/weather_336/fake_weather_336.csv'
]
names = [
    # '0_etth1_96','0_etth1_192','etth1_336',
    # '0_etth2_96','0_etth2_192','etth2_336',
    # 'tec_96','tec_192','tec_336'
    'weather_96','weather_192','weather_336',
]
pred_len = [
    # 96,192,336,
    # 96,192,336,
    96,192,336,
]

# 读取完整数据
for i in range(len(ori)):
    
    ori_df = pd.read_csv(ori[i])
    
    # 删除date列（假设date是第一列或列名为'date'）
    if 'date' in ori_df.columns:
        ori_df = ori_df.drop('date', axis=1)
    else:
        ori_df = ori_df.iloc[:, 1:]  # 删除第一列
    
    ori_data = ori_df.values
    print('1 load: ori_data shape =', ori_data.shape)
    
    # fake_data = pd.read_csv(gen[i]).values
    # print('2 load: fake_data shape =', fake_data.shape)
    
    seq_len = 512+pred_len[i]
    num_features = ori_data.shape[1] 
    
    # 重新组织数据
    ori_data = ori_data.reshape(-1, seq_len, num_features)
    # fake_data = fake_data.reshape(-1, seq_len, num_features)
    fake_data = None

    name = names[i]
    visualization(name=name, ori_data=ori_data, generated_data=fake_data, analysis='pca', compare=ori_data.shape[0])
    visualization(name=name, ori_data=ori_data, generated_data=fake_data, analysis='tsne', compare=ori_data.shape[0])
    visualization(name=name, ori_data=ori_data, generated_data=fake_data, analysis='kernel', compare=ori_data.shape[0])
