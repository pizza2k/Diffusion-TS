import pandas as pd
from Utils.metric_utils import visualization

print('start')

# 读取完整数据
ori_df = pd.read_csv('Data/datasets/tail_samples_2014.csv')

# 删除date列（假设date是第一列或列名为'date'）
if 'date' in ori_df.columns:
    ori_df = ori_df.drop('date', axis=1)
else:
    ori_df = ori_df.iloc[:, 1:]  # 删除第一列

ori_data = ori_df.values
print('1 load: ori_data shape =', ori_data.shape)

fake_data = pd.read_csv('OUTPUT/tail_2014/ddpm_fake_tail_2014.csv').values
print('2 load: fake_data shape =', fake_data.shape)

seq_len = 704
num_features = ori_data.shape[1] 

# 重新组织数据
ori_data = ori_data.reshape(-1, seq_len, num_features)
fake_data = fake_data.reshape(-1, seq_len, num_features)

visualization(ori_data=ori_data, generated_data=fake_data, analysis='pca', compare=ori_data.shape[0])
visualization(ori_data=ori_data, generated_data=fake_data, analysis='tsne', compare=ori_data.shape[0])
visualization(ori_data=ori_data, generated_data=fake_data, analysis='kernel', compare=ori_data.shape[0])
