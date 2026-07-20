import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class ResBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        negative_slope=0.01
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.BatchNorm1d(
                out_dim
            ),
            nn.LeakyReLU(
                negative_slope=negative_slope
            )
        )

    def forward(self, x):
        return self.conv(x) + x


class Generator(nn.Module):
    def __init__(
        self,
        node_feat_dim
    ):

        super().__init__()

        self.raw_node_feat_dim = node_feat_dim

        self.negative_slope = 0.01


        self.gat_heads = 2

        self.model_node_feat_dim = (
            (
                node_feat_dim
                + self.gat_heads
                - 1
            )
            // self.gat_heads
        ) * self.gat_heads

        self.pad_dim = (
            self.model_node_feat_dim
            - self.raw_node_feat_dim
        )

        gat_out_channels = (
            self.model_node_feat_dim
            // self.gat_heads
        )

        # ========================================================
        # 3 层 GAT 特征提取
        # ========================================================

        self.gat1 = gnn.GATConv(
            self.model_node_feat_dim,
            gat_out_channels,
            heads=self.gat_heads,
            concat=True,
            dropout=0.1,
            add_self_loops=True
        )

        self.bn1 = nn.BatchNorm1d(
            self.model_node_feat_dim
        )

        self.res1 = ResBlock(
            self.model_node_feat_dim,
            self.model_node_feat_dim
        )

        self.gat2 = gnn.GATConv(
            self.model_node_feat_dim,
            gat_out_channels,
            heads=self.gat_heads,
            concat=True,
            dropout=0.1,
            add_self_loops=True
        )

        self.bn2 = nn.BatchNorm1d(
            self.model_node_feat_dim
        )

        self.res2 = ResBlock(
            self.model_node_feat_dim,
            self.model_node_feat_dim
        )

        self.gat3 = gnn.GATConv(
            self.model_node_feat_dim,
            self.model_node_feat_dim,
            heads=1,
            concat=False,
            dropout=0.1,
            add_self_loops=True
        )

        self.bn3 = nn.BatchNorm1d(
            self.model_node_feat_dim
        )

        self.res3 = ResBlock(
            self.model_node_feat_dim,
            self.model_node_feat_dim
        )

        self.mu_esu_branch = nn.Sequential(
            nn.Linear(
                self.model_node_feat_dim,
                self.model_node_feat_dim * 2
            ),

            nn.BatchNorm1d(
                self.model_node_feat_dim * 2
            ),

            nn.LeakyReLU(
                negative_slope=self.negative_slope
            ),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                self.model_node_feat_dim * 2,
                self.model_node_feat_dim
            ),

            nn.BatchNorm1d(
                self.model_node_feat_dim
            ),

            nn.LeakyReLU(
                negative_slope=self.negative_slope
            ),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                self.model_node_feat_dim,
                self.model_node_feat_dim
            ),

            nn.BatchNorm1d(
                self.model_node_feat_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                self.model_node_feat_dim,
                self.raw_node_feat_dim
            ),

            nn.Softplus()
        )

        self.s_esu_branch = nn.Sequential(
            nn.Linear(
                self.model_node_feat_dim,
                self.model_node_feat_dim * 2
            ),

            nn.BatchNorm1d(
                self.model_node_feat_dim * 2
            ),

            nn.LeakyReLU(
                negative_slope=self.negative_slope
            ),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                self.model_node_feat_dim * 2,
                self.model_node_feat_dim
            ),

            nn.BatchNorm1d(
                self.model_node_feat_dim
            ),

            nn.LeakyReLU(
                negative_slope=self.negative_slope
            ),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                self.model_node_feat_dim,
                self.model_node_feat_dim
            ),

            nn.BatchNorm1d(
                self.model_node_feat_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                self.model_node_feat_dim,
                self.raw_node_feat_dim
            )
        )

    def _pad_input_feature(self, x_flat):
        feat_dim = x_flat.size(-1)

        if feat_dim == self.model_node_feat_dim:
            return x_flat

        if feat_dim == self.raw_node_feat_dim:
            if self.pad_dim > 0:
                x_flat = F.pad(
                    x_flat,
                    (0, self.pad_dim),
                    value=0.0
                )

            return x_flat

        raise ValueError(
            f"输入特征维度不匹配："
            f"当前输入维度为 {feat_dim}，"
        )

    def forward(self, x, edge_index):
        if x.dim() == 3 and x.size(0) == 1:
            x_flat = x.squeeze(0)

        elif x.dim() == 2:
            x_flat = x

        else:
            raise ValueError(
                f"不支持的 x 形状：{x.shape}，"
            )

        x_flat = self._pad_input_feature(
            x_flat
        )

        # ========================================================
        # 第一层 GAT
        # ========================================================

        gat1_out = self.gat1(
            x_flat,
            edge_index,
        )

        gat1_out = self.bn1(
            gat1_out
        )

        x1 = (
            F.leaky_relu(
                gat1_out,
                negative_slope=self.negative_slope
            )
            + self.res1(x_flat)
        )

        x1 = F.dropout(
            x1,
            p=0.1,
            training=self.training
        )

        # ========================================================
        # 第二层 GAT
        # ========================================================

        gat2_out = self.gat2(
            x1,
            edge_index,
        )

        gat2_out = self.bn2(
            gat2_out
        )

        x2 = (
            F.leaky_relu(
                gat2_out,
                negative_slope=self.negative_slope
            )
            + self.res2(x1)
        )

        x2 = F.dropout(
            x2,
            p=0.1,
            training=self.training
        )

        # ========================================================
        # 第三层 GAT
        # ========================================================

        gat3_out = self.gat3(
            x2,
            edge_index,
        )

        gat3_out = self.bn3(
            gat3_out
        )

        x3 = (
            F.leaky_relu(
                gat3_out,
                negative_slope=self.negative_slope
            )
            + self.res3(x2)
        )

        # ========================================================
        # 双分支输出
        # ========================================================

        mu_esu = self.mu_esu_branch(
            x3
        )

        s_esu = self.s_esu_branch(
            x3
        )

        return (
            mu_esu.unsqueeze(0),
            s_esu.unsqueeze(0)
        )


__all__ = [
    "ResBlock",
    "Generator",
]