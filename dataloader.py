import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, df, max_seq_len, user_feature_cols, ad_feature_cols, label_col):
        """
        :param df: 原始 DataFrame
        :param max_seq_len: 最大序列长度，用于 padding 序列
        :param user_feature_cols: 用户特征列名列表
        :param ad_feature_cols: 广告特征列名列表
        # :param seq_feature_col: 序列特征列名
        :param label_col: 标签列名
        """
        self.df = df
        self.max_seq_len = max_seq_len
        self.user_feature_cols = user_feature_cols
        self.ad_feature_cols = ad_feature_cols
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 用户特征
        user_features = torch.tensor(row[self.user_feature_cols], dtype=torch.float)
        
        # 广告特征
        ad_features = torch.tensor(row[self.ad_feature_cols], dtype=torch.float)
        
        # 序列特征 (需要 padding)
        btag_hist = self.pad_sequence(row['btag_hist'], self.max_seq_len, padding_value=0, dtype=torch.int64)
        cate_hist = self.pad_sequence(row['cate_hist'], self.max_seq_len, padding_value=0, dtype=torch.int64)
        brand_hist = self.pad_sequence(row['brand_hist'], self.max_seq_len, padding_value=0, dtype=torch.int64)
        time_hist = self.pad_sequence(row['time_hist'], self.max_seq_len, padding_value=0.0, dtype=torch.int64)
        
        # 标签
        label = torch.tensor(row[self.label_col], dtype=torch.float)
        #
        
        return user_features, ad_features, (btag_hist, cate_hist,brand_hist,time_hist), label

    def pad_sequence(self, sequence, max_len, padding_value, dtype):
        """
        对输入序列进行padding。
        - sequence: 输入的序列 (list)
        - max_len: 需要padding的最大长度
        - padding_value: 用于填充的值 (默认: 0)
        - dtype: 转换后的数据类型 (默认: torch.int64)
        """
        seq_len = len(sequence)
        # 如果序列长度超过 max_len，进行截断
        if seq_len > max_len:
            sequence = sequence[:max_len]
        else:
            # 否则进行 padding
            sequence = sequence + [padding_value] * (max_len - seq_len)

        # 转换为Tensor
        padded_seq = torch.tensor(sequence, dtype=dtype)
        return padded_seq
