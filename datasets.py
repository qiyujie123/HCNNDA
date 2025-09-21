# datasets.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch


def load_lncRNA_drug_data(config):
    """加载CSV格式的生物数据
    参数：
    - config: 包含数据路径配置的字典，需要包含：
        data_dir: 数据目录路径
        lncRNA_view_files: lncRNA相似性矩阵文件名列表
        drug_view_files: 药物相似性矩阵文件名列表
        adj_file: 关联矩阵文件名
    """
    # 初始化容器
    lnc_views = []
    drug_views = []

    # 1. 加载lncRNA相似性矩阵 -------------------------------------------------
    for fname in config['lncRNA_view_files']:
        file_path = os.path.join(config['data_dir'], fname)

        # 异常处理：文件存在性检查
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"lncRNA相似性文件 {fname} 未找到")

        # 读取CSV
        df = pd.read_csv(file_path, index_col=0)

        # 转换为numpy数组并归一化
        sim_matrix = df.values.astype(np.float32)
        sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix) + 1e-8)

        lnc_views.append(sim_matrix)

    # 2. 加载药物相似性矩阵 ---------------------------------------------------
    for fname in config['drug_view_files']:
        file_path = os.path.join(config['data_dir'], fname)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"药物相似性文件 {fname} 未找到")

        df = pd.read_csv(file_path, index_col=0)
        sim_matrix = df.values.astype(np.float32)
        sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix) + 1e-8)
        drug_views.append(sim_matrix)
        # 3. 加载关联矩阵 --------------------------------------------------------
        adj_path = os.path.join(config['data_dir'], config['adj_file'])
        if not os.path.exists(adj_path):
            raise FileNotFoundError(f"关联矩阵文件 {config['adj_file']} 未找到")

    adj_df = pd.read_csv(adj_path, index_col=0)
    adj_matrix = adj_df.values.astype(np.float32)

    # 4. 验证数据一致性 ------------------------------------------------------
    # 检查lncRNA数量一致
    lnc_counts = [m.shape[0] for m in lnc_views]
    if len(set(lnc_counts)) > 1:
        raise ValueError(f"lncRNA视图维度不一致: {lnc_counts}")
    if lnc_counts[0] != adj_matrix.shape[0]:
        raise ValueError(f"lncRNA数量不匹配: 视图维度{lnc_counts[0]} vs 关联矩阵行数{adj_matrix.shape[0]}")

    # 检查药物数量一致
    drug_counts = [m.shape[0] for m in drug_views]
    if len(set(drug_counts)) > 1:
        raise ValueError(f"药物视图维度不一致: {drug_counts}")
    if drug_counts[0] != adj_matrix.shape[1]:
        raise ValueError(f"药物数量不匹配: 视图维度{drug_counts[0]} vs 关联矩阵列数{adj_matrix.shape[1]}")

    # 转换为PyTorch张量
    lnc_views = [torch.FloatTensor(m) for m in lnc_views]
    drug_views = [torch.FloatTensor(m) for m in drug_views]
    adj_matrix = torch.FloatTensor(adj_matrix)

    return lnc_views, drug_views, adj_matrix