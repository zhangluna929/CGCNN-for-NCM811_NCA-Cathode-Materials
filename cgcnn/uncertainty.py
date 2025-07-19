"""
Uncertainty Quantification for CGCNN

Bayesian neural networks and ensemble methods for uncertainty estimation
in materials property prediction. Includes MC-Dropout, variational inference,
and calibration techniques.

Author: lunazhang
Date: 2023
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from .model import CrystalGraphConvNet, ConvLayer


class BayesianLinear(nn.Module):
    """
    贝叶斯线性层
    Bayesian Linear Layer with weight uncertainty
    """
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # 权重参数的均值和方差
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 偏置参数的均值和方差
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -3)
        nn.init.uniform_(self.bias_mu, -0.1, 0.1)
        nn.init.constant_(self.bias_rho, -3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，采样权重"""
        # 计算权重标准差
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        # 采样权重和偏置
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_std * weight_eps
        bias = self.bias_mu + bias_std * bias_eps
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """计算KL散度"""
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        # 权重的KL散度
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu.pow(2) + weight_std.pow(2)) / self.prior_std**2 
            - 2 * torch.log(weight_std / self.prior_std) - 1
        )
        
        # 偏置的KL散度
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu.pow(2) + bias_std.pow(2)) / self.prior_std**2 
            - 2 * torch.log(bias_std / self.prior_std) - 1
        )
        
        return weight_kl + bias_kl


class BayesianCGCNN(nn.Module):
    """
    贝叶斯图卷积神经网络
    Bayesian Crystal Graph Convolutional Neural Network
    """
    
    def __init__(self, orig_atom_fea_len: int, nbr_fea_len: int,
                 atom_fea_len: int = 64, n_conv: int = 3, h_fea_len: int = 128, 
                 n_h: int = 1, classification: bool = False, prior_std: float = 1.0):
        super(BayesianCGCNN, self).__init__()
        self.classification = classification
        self.prior_std = prior_std
        
        # 确定性层
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        # 贝叶斯全连接层
        if n_h > 1:
            self.fcs = nn.ModuleList([BayesianLinear(h_fea_len, h_fea_len, prior_std)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)])
        
        # 输出层
        if self.classification:
            self.fc_out = BayesianLinear(h_fea_len, 2, prior_std)
            self.logsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = BayesianLinear(h_fea_len, 1, prior_std)
        
        if self.classification:
            self.dropout = nn.Dropout()
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """前向传播"""
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out
    
    def pooling(self, atom_fea, crystal_atom_idx):
        """池化操作"""
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
    
    def kl_divergence(self) -> torch.Tensor:
        """计算总KL散度"""
        kl_sum = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl_sum += module.kl_divergence()
        return kl_sum
    
    def predict_with_uncertainty(self, *args, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        不确定性预测
        
        Returns:
            mean: 预测均值
            std: 预测标准差
        """
        self.train()  # 启用dropout
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(*args)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return mean, std


class EnsembleCGCNN(nn.Module):
    """
    集成学习CGCNN
    Ensemble CGCNN for uncertainty estimation
    """
    
    def __init__(self, models: List[CrystalGraphConvNet]):
        super(EnsembleCGCNN, self).__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
    
    def forward(self, *args):
        """前向传播，返回所有模型的预测"""
        predictions = []
        for model in self.models:
            pred = model(*args)
            predictions.append(pred)
        return torch.stack(predictions)
    
    def predict_with_uncertainty(self, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        集成预测与不确定性估计
        
        Returns:
            mean: 预测均值
            std: 预测标准差（认识不确定性）
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(*args)  # [n_models, batch_size, output_dim]
            mean = torch.mean(predictions, dim=0)
            std = torch.std(predictions, dim=0)
        
        return mean, std
    
    def predict_with_aleatoric_uncertainty(self, *args, n_dropout: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测总不确定性（认识+偶然）
        
        Returns:
            mean: 预测均值
            epistemic_std: 认识不确定性
            aleatoric_std: 偶然不确定性
        """
        self.train()  # 启用dropout
        all_predictions = []
        
        for model in self.models:
            model_predictions = []
            for _ in range(n_dropout):
                with torch.no_grad():
                    pred = model(*args)
                    model_predictions.append(pred)
            all_predictions.extend(model_predictions)
        
        all_predictions = torch.stack(all_predictions)
        
        # 总均值和方差
        total_mean = torch.mean(all_predictions, dim=0)
        total_var = torch.var(all_predictions, dim=0)
        
        # 每个模型内部的方差（偶然不确定性）
        within_model_vars = []
        for i in range(self.n_models):
            start_idx = i * n_dropout
            end_idx = (i + 1) * n_dropout
            model_preds = all_predictions[start_idx:end_idx]
            within_var = torch.var(model_preds, dim=0)
            within_model_vars.append(within_var)
        
        aleatoric_var = torch.mean(torch.stack(within_model_vars), dim=0)
        epistemic_var = total_var - aleatoric_var
        
        return total_mean, torch.sqrt(epistemic_var), torch.sqrt(aleatoric_var)


class CalibrationModule:
    """
    置信度校准模块
    Confidence Calibration Module
    """
    
    def __init__(self, method: str = 'temperature_scaling'):
        self.method = method
        self.calibrator = None
    
    def fit(self, logits: torch.Tensor, targets: torch.Tensor):
        """训练校准器"""
        if self.method == 'temperature_scaling':
            self.calibrator = self._fit_temperature_scaling(logits, targets)
        elif self.method == 'platt_scaling':
            self.calibrator = self._fit_platt_scaling(logits, targets)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
    
    def _fit_temperature_scaling(self, logits: torch.Tensor, targets: torch.Tensor) -> nn.Module:
        """温度缩放校准"""
        class TemperatureScaling(nn.Module):
            def __init__(self):
                super().__init__()
                self.temperature = nn.Parameter(torch.ones(1))
            
            def forward(self, logits):
                return logits / self.temperature
        
        temp_model = TemperatureScaling()
        optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = temp_model(logits)
            loss = F.cross_entropy(scaled_logits, targets)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return temp_model
    
    def _fit_platt_scaling(self, logits: torch.Tensor, targets: torch.Tensor) -> nn.Module:
        """Platt缩放校准"""
        class PlattScaling(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)
            
            def forward(self, logits):
                return torch.sigmoid(self.linear(logits))
        
        platt_model = PlattScaling(logits.size(-1))
        optimizer = torch.optim.Adam(platt_model.parameters(), lr=0.001)
        
        for _ in range(100):
            optimizer.zero_grad()
            probs = platt_model(logits).squeeze()
            loss = F.binary_cross_entropy(probs, targets.float())
            loss.backward()
            optimizer.step()
        
        return platt_model
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """应用校准"""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted")
        
        with torch.no_grad():
            return self.calibrator(logits)
    
    def expected_calibration_error(self, probs: torch.Tensor, targets: torch.Tensor, 
                                 n_bins: int = 10) -> float:
        """计算期望校准误差(ECE)"""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower.item()) & (probs <= bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = targets[in_bin].float().mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()


class UncertaintyMetrics:
    """
    不确定性评估指标
    Uncertainty Evaluation Metrics
    """
    
    @staticmethod
    def mutual_information(predictions: torch.Tensor) -> torch.Tensor:
        """
        计算互信息（总不确定性）
        
        Args:
            predictions: [n_samples, batch_size, n_classes]
        """
        # 平均预测概率
        mean_probs = torch.mean(predictions, dim=0)
        
        # 互信息 = H(y) - E[H(y|θ)]
        # H(y) = -sum(p * log(p))
        entropy_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        # E[H(y|θ)] = E[-sum(p_θ * log(p_θ))]
        sample_entropies = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1)
        mean_entropy = torch.mean(sample_entropies, dim=0)
        
        mutual_info = entropy_mean - mean_entropy
        return mutual_info
    
    @staticmethod
    def predictive_entropy(predictions: torch.Tensor) -> torch.Tensor:
        """计算预测熵"""
        mean_probs = torch.mean(predictions, dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        return entropy
    
    @staticmethod
    def variance_ratio(predictions: torch.Tensor) -> torch.Tensor:
        """计算方差比率"""
        mean_probs = torch.mean(predictions, dim=0)
        max_prob = torch.max(mean_probs, dim=-1)[0]
        return 1 - max_prob


def create_ensemble_from_checkpoints(checkpoint_paths: List[str], model_config: dict) -> EnsembleCGCNN:
    """
    从检查点创建集成模型
    
    Args:
        checkpoint_paths: 模型检查点路径列表
        model_config: 模型配置参数
    
    Returns:
        集成模型
    """
    models = []
    for path in checkpoint_paths:
        model = CrystalGraphConvNet(**model_config)
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        models.append(model)
    
    return EnsembleCGCNN(models)


# 使用示例
def example_usage():
    """使用示例"""
    # 创建贝叶斯CGCNN
    bayesian_model = BayesianCGCNN(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=2,
        classification=False,
        prior_std=1.0
    )
    
    # 贝叶斯训练损失函数
    def bayesian_loss(model, output, target, kl_weight=1e-6):
        mse_loss = F.mse_loss(output, target)
        kl_loss = model.kl_divergence()
        return mse_loss + kl_weight * kl_loss
    
    # 不确定性预测
    # mean, std = bayesian_model.predict_with_uncertainty(
    #     atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, n_samples=100
    # ) 