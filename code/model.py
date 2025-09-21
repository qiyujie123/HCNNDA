import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalConsensusPredictor(nn.Module):
    def __init__(self,
                 lncRNA_dims,
                 drug_dims,
                 hidden_dim=64,
                 pred_dim=32,
                 disable_coding=False,
                 disable_class=False,
                 disable_global=False):
        super(HierarchicalConsensusPredictor, self).__init__()

        # 配置开关
        self.disable_coding = disable_coding
        self.disable_class = disable_class
        self.disable_global = disable_global

        # === 1. 多视图编码器 ===
        self.lnc_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ) for d in lncRNA_dims
        ])
        self.drug_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ) for d in drug_dims
        ])

        # === 2. 视图注意力机制 ===
        self.view_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

        # === 3. 分类共识编码器 ===
        self.class_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2),
            num_layers=2
        )
        self.class_norm = nn.LayerNorm(hidden_dim)

        self.drug_class_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2),
            num_layers=2
        )
        self.drug_class_norm = nn.LayerNorm(hidden_dim)

        # === 4. 编码共识模块 ===
        self.code_consensus = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # === 5. 全局共识模块 ===
        self.global_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # === 6. 门控与预测器 ===
        self.gate_layer = nn.Linear(hidden_dim * 3, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, pred_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(pred_dim, 1),
            nn.Sigmoid()
        )

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, lnc_views, drug_views, return_feature=False):
        # === 1. 特征编码 ===
        lnc_embs = [enc(view) for enc, view in zip(self.lnc_encoders, lnc_views)]
        drug_embs = [enc(view) for enc, view in zip(self.drug_encoders, drug_views)]

        # === 2. 视图注意力聚合 ===
        attn_out_lnc, _ = self.view_attn(torch.stack(lnc_embs, dim=0),
                                         torch.stack(lnc_embs, dim=0),
                                         torch.stack(lnc_embs, dim=0))
        avg_lnc_emb = attn_out_lnc.mean(dim=0)

        attn_out_drug, _ = self.view_attn(torch.stack(drug_embs, dim=0),
                                          torch.stack(drug_embs, dim=0),
                                          torch.stack(drug_embs, dim=0))
        avg_drug_emb = attn_out_drug.mean(dim=0)

        # === 3. 分类共识 ===
        if not self.disable_class:
            class_attn = self.class_encoder(avg_lnc_emb.unsqueeze(0)).squeeze(0)
            class_attn = self.class_norm(class_attn)
            drug_class_attn = self.drug_class_encoder(avg_drug_emb.unsqueeze(0)).squeeze(0)
            drug_class_attn = self.drug_class_norm(drug_class_attn)
            combined_class = torch.cat([class_attn, drug_class_attn], dim=-1)
        else:
            combined_class = torch.cat([avg_lnc_emb, avg_drug_emb], dim=-1)

        # === 4. 编码共识（对比学习） ===
        if not self.disable_coding:
            code_loss = 0.0
            for lnc_emb, drug_emb in zip(lnc_embs, drug_embs):
                lnc_c = self.code_consensus(lnc_emb)
                drug_c = self.code_consensus(drug_emb)
                # 正样本
                pos_loss = F.cosine_embedding_loss(
                    lnc_c, drug_c, torch.ones(lnc_c.size(0), device=lnc_c.device))
                # 负样本（随机打乱）
                shuffled = drug_c[torch.randperm(drug_c.size(0))]
                neg_loss = F.cosine_embedding_loss(
                    lnc_c, shuffled, -torch.ones(lnc_c.size(0), device=lnc_c.device))
                code_loss += pos_loss + 0.5 * neg_loss
        else:
            code_loss = torch.tensor(0.0, device=avg_lnc_emb.device)

        # === 5. 全局共识 ===
        if not self.disable_global:
            global_emb_lncrna, _ = self.global_attn(avg_lnc_emb.unsqueeze(0),
                                                    avg_drug_emb.unsqueeze(0),
                                                    avg_drug_emb.unsqueeze(0))
            global_emb_lncrna = self.global_proj(global_emb_lncrna.squeeze(0))

            global_emb_drug, _ = self.global_attn(avg_drug_emb.unsqueeze(0),
                                                  avg_lnc_emb.unsqueeze(0),
                                                  avg_lnc_emb.unsqueeze(0))
            global_emb_drug = self.global_proj(global_emb_drug.squeeze(0))

            global_emb = (global_emb_lncrna + global_emb_drug) / 2
            global_loss = F.mse_loss(avg_lnc_emb, avg_drug_emb)
        else:
            global_emb = torch.zeros_like(avg_lnc_emb)
            global_loss = torch.tensor(0.0, device=avg_lnc_emb.device)

        # === 6. 门控机制与预测 ===
        fused_input = torch.cat([combined_class, global_emb], dim=-1)
        gate = torch.sigmoid(self.gate_layer(fused_input))
        final_feature = torch.cat([gate, global_emb], dim=-1)
        pred = self.predictor(final_feature)

        # 返回结果
        if return_feature:
            return final_feature.detach()

        return pred, {
            'coding': code_loss,
            'global': global_loss
        }
