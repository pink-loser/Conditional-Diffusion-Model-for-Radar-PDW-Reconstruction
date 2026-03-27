import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm

# 导入你提供的 1D 库组件
from ddpm1d import Unet1D, GaussianDiffusion1D, Transformer1D

# ==========================================
# 1. 数据预处理类 (保持逻辑，输出原始网格)
# ==========================================
class PDWPreprocessor:
    def __init__(self, seq_len=4096, dt=1.0, threshold_ratio=0.4):
        self.seq_len = seq_len
        self.dt = dt
        self.threshold_ratio = threshold_ratio

    def encode(self, pri_array, rf_array, pw_array):
        """
        将 PDW 转换为以时间步作为索引的二维网格
        """
        # 1. 计算 TOA (用于索引)
        toa_array = np.cumsum(np.insert(pri_array[:-1], 0, 0))
        
        # 2. 初始化网格: [PRI, RF, PW, Mask]
        grid = np.zeros((self.seq_len, 4), dtype=float)
        
        bins = {}
        for i in range(len(toa_array)):
            idx = int(toa_array[i] // self.dt)
            if idx >= self.seq_len: break
                
            if idx not in bins: bins[idx] = []
            # 存入所有参数
            bins[idx].append({
                'pri': pri_array[i],    # pri 被分给了前一个脉冲
                'rf': rf_array[i], 
                'pw': pw_array[i]
            })

        # 3. 处理冲突
        for idx, pulses in bins.items():
            if len(pulses) == 1:
                # 只有一个脉冲，直接存入
                p = pulses[0]
                grid[idx] = [p['pri'], p['rf'], p['pw'], 1.0]
            else:
                # 1. 确定衡量标准（这里建议用 PW 脉宽或 PRI，根据你的需求决定）
                # 我们先找出这一组脉冲里的最大值
                max_val = max(p['pw'] for p in pulses) # 假设以脉宽 PW 作为衡量大小的标准
                
                # 2. 筛选：只保留那些“足够大”的脉冲 (即：值 > 最大值 / 阈值)
                # 比如最大 PW 是 100，阈值是 2.0，那么低于 50 的脉冲都会被直接删掉
                qualified_pulses = [p for p in pulses if p['pw'] >= max_val * self.threshold_ratio]
                
                # 3. 对筛选后的结果进行处理
                if len(qualified_pulses) == 1:
                    # 只有 1 个胜出，直接存入
                    p = qualified_pulses[0]
                    grid[idx] = [p['pri'], p['rf'], p['pw'], 1.0]
                else:
                    # 有多个脉冲大小接近（且都通过了筛选），进行加权合并
                    total_weight = sum(p['pw'] for p in qualified_pulses) + 1e-9 # 防止除零
                    
                    avg_pri = sum(p['pri'] * p['pw'] for p in qualified_pulses) / total_weight
                    avg_rf  = sum(p['rf'] * p['pw'] for p in qualified_pulses) / total_weight
                    avg_pw  = sum(p['pw'] for p in qualified_pulses) / len(qualified_pulses)
                    
                    grid[idx] = [avg_pri, avg_rf, avg_pw, 1.0]

        return grid

# ==========================================
# 2. 1D 多通道数据集 (包含对数归一化)
# ==========================================
class MultiModeRadarDataset1D(Dataset):
    def __init__(self, pkl_folder, seq_len=4096, clip_sigma=3.0):
        self.all_samples = []
        self.seq_len = seq_len
        self.clip_sigma = clip_sigma
        self.preprocessor = PDWPreprocessor(seq_len=seq_len)
        
        for file_name in os.listdir(pkl_folder):
            if file_name.endswith('.pkl'):
                with open(os.path.join(pkl_folder, file_name), 'rb') as f:
                    self.all_samples.extend(pickle.load(f))
        
        # 基础统计值
        # 2. 统计 RF 的全局均值和标准差 (建议从干净数据中统计)
        all_rf = []
        all_pw = []
        for s in self.all_samples:
            all_rf.extend(s['clean']['rf'])
            all_pw.extend(s['clean']['pw']) # 新增：统计 PW
        
        self.rf_mean = np.mean(all_rf)
        self.rf_std = np.std(all_rf) + 1e-6
        
        # 2. 新增：计算 PW 的全局均值和标准差
        self.pw_mean = np.mean(all_pw)
        self.pw_std = np.std(all_pw) + 1e-6
        
        self.stats = {
            'pri': {'min': 0, 'max': 100}
            # 'pw':  {'min': 0, 'max': 20}
        }
        # 计算 RF 的 Log 统计值用于归一化
        # self.log_rf_min = np.log1p(self.stats['rf']['min'])
        # self.log_rf_max = np.log1p(self.stats['rf']['max'])

    def normalize(self, grid):
        g = grid.copy()
        # PRI: 线性归一化
        g[:, 0] = ((g[:, 0] - self.stats['pri']['min']) / (self.stats['pri']['max'] - self.stats['pri']['min'])) * 2 - 1
        
        # RF: Standardization + Clipping
        # 1. Z-score
        rf_z = (g[:, 1] - self.rf_mean) / self.rf_std
        # 2. Clipping 到 [-3, 3]
        rf_clipped = np.clip(rf_z, -self.clip_sigma, self.clip_sigma)
        # 3. 映射到 [-1, 1]
        g[:, 1] = rf_clipped / self.clip_sigma
        
        # PW: 线性归一化
        # g[:, 2] = ((g[:, 2] - self.stats['pw']['min']) / (self.stats['pw']['max'] - self.stats['pw']['min'])) * 2 - 1
        pw_z = (g[:, 2] - self.pw_mean) / self.pw_std
        pw_clipped = np.clip(pw_z, -self.clip_sigma, self.clip_sigma)
        g[:, 2] = pw_clipped / self.clip_sigma
        
        # Mask: [0, 1] -> [-1, 1]
        g[:, 3] = g[:, 3] * 2 - 1
        return g

    def denormalize(self, grid_1d):
        """将 (4, 4096) 的 1D 数组还原为物理值"""
        g = grid_1d.copy().T # 转回 (4096, 4)
        
        # PRI 反归一化
        g[:, 0] = (g[:, 0] + 1) / 2 * (self.stats['pri']['max'] - self.stats['pri']['min']) + self.stats['pri']['min']
        
        # RF 反归一化
        # 1. 从 [-1, 1] 还原回 [-3, 3]
        rf_clipped = g[:, 1] * self.clip_sigma
        # 2. 还原物理值 (超出裁剪范围的部分会丢失，但在雷达参数中通常不重要)
        g[:, 1] = rf_clipped * self.rf_std + self.rf_mean
        
        # PW 反归一化
        # g[:, 2] = (g[:, 2] + 1) / 2 * (self.stats['pw']['max'] - self.stats['pw']['min']) + self.stats['pw']['min']
        pw_clipped = g[:, 2] * self.clip_sigma
        g[:, 2] = pw_clipped * self.pw_std + self.pw_mean
        
        # Mask 反归一化
        g[:, 3] = (g[:, 3] + 1) / 2
        return g

    def to_tensor(self, grid):
        # 1D 结构: [Channels, Length] -> [8, 4096]
        padded = np.zeros((8, self.seq_len), dtype=np.float32)
        padded[:4, :] = grid.T # 将 (4096, 4) 转置为 (4, 4096)
        return torch.from_numpy(padded)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        sample = self.all_samples[index]
        clean_grid = self.normalize(self.preprocessor.encode(sample['clean']['pri'], sample['clean']['rf'], sample['clean']['pw']))
        corr_grid = self.normalize(self.preprocessor.encode(sample['corrupted']['pri'], sample['corrupted']['rf'], sample['corrupted']['pw']))
        return self.to_tensor(clean_grid), self.to_tensor(corr_grid)

# ==========================================
# 3. 绘图代码 (只显示非零值)
# ==========================================
def plot_repair_result_1d(res_clean, res_corr, res_repair, mode_name, step_name):
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    channels = ['PRI', 'RF', 'PW', 'Mask']
    colors = ['blue', 'green', 'orange', 'red']
    time_steps = np.arange(res_clean.shape[0])
    
    # 修复结果的置信度掩码
    mask_threshold = 0.5
    valid_repair_mask = res_repair[:, 3] > mask_threshold

    for i in range(4):
        # 过滤非零点
        m_clean = res_clean[:, i] > 1e-6
        m_corr = res_corr[:, i] > 1e-6
        
        # 1. Target
        axes[i].scatter(time_steps[m_clean], res_clean[m_clean, i], color=colors[i], alpha=0.2, s=15, label='Target')
        # 2. Corrupted
        axes[i].scatter(time_steps[m_corr], res_corr[m_corr, i], color='black', marker='x', alpha=0.2, s=20, label='Corrupted')
        
        # 3. Repaired (仅绘制有效点)
        if i < 3:
            m_repair = valid_repair_mask & (res_repair[:, i] > 1e-6)
            axes[i].scatter(time_steps[m_repair], res_repair[m_repair, i], color=colors[i], edgecolors='white', linewidths=0.5, s=25, label='Repaired')
        else:
            # Mask 通道固定坐标轴，防止视觉缩放
            m_repair = res_repair[:, i] > 1e-6
            axes[i].scatter(time_steps[m_repair], res_repair[m_repair, i], color=colors[i], s=10, label='Confidence')
            axes[i].set_ylim(-0.1, 1.1)

        axes[i].set_ylabel(channels[i])
        axes[i].grid(True, linestyle=':', alpha=0.5)
        axes[i].legend(loc='upper right')

    plt.suptitle(f'Radar PDW 1D Repair | Mode: {mode_name} | Step: {step_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'repair_{mode_name}_step_{step_name}.png', dpi=150)
    plt.close()

@torch.no_grad()
def evaluate_multi_datasets(model, diffusion, dataset, device, step_name, folder='./radar_datasets', num_datasets=5):
    """
    从数据集中随机抽取 5 个不同的模式进行修复并绘图
    """
    model.eval()
    all_pkls = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    # 确保只选出最多 5 个数据集
    selected_pkls = np.random.choice(all_pkls, min(num_datasets, len(all_pkls)), replace=False)
    
    for f_name in selected_pkls:
        mode_name = f_name.replace('pdw_dataset_', '').replace('.pkl', '')
        with open(os.path.join(folder, f_name), 'rb') as f:
            mode_data = pickle.load(f)
        
        # 1. 随机取一个样本并转换
        sample = mode_data[np.random.randint(len(mode_data))]
        # 利用 dataset 的预处理逻辑
        clean_grid = dataset.normalize(dataset.preprocessor.encode(sample['clean']['pri'], sample['clean']['rf'], sample['clean']['pw']))
        corr_grid = dataset.normalize(dataset.preprocessor.encode(sample['corrupted']['pri'], sample['corrupted']['rf'], sample['corrupted']['pw']))
        
        clean_t = dataset.to_tensor(clean_grid).unsqueeze(0).to(device) # [1, 8, 4096]
        corr_t = dataset.to_tensor(corr_grid).unsqueeze(0).to(device)
        
        # 2. 1000步 DDPM 采样
        img = torch.randn_like(clean_t)
        for t in tqdm(reversed(range(0, diffusion.num_timesteps)), desc=f'Evaluating {mode_name}', leave=False):
            img, _ = diffusion.p_sample(img, t, x_self_cond=corr_t)
            
        # 3. 反归一化
        res_clean = dataset.denormalize(clean_t[0].cpu().numpy()[:4])
        res_corr = dataset.denormalize(corr_t[0].cpu().numpy()[:4])
        res_repair = dataset.denormalize(img[0].cpu().numpy()[:4])
        
        # 4. 调用你提供的绘图函数
        plot_repair_result_1d(res_clean, res_corr, res_repair, mode_name, step_name)
    
    model.train()

def plot_loss(loss_history, save_path='loss_curve_1d.png'):
    """绘制训练过程中的 Loss 曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Train MSE Loss (v-pred)', color='purple', alpha=0.7)
    plt.yscale('log') # 建议使用对数轴，方便观察后期细微的收敛
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.title('1D PDW Diffusion Training Loss')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to: {save_path}")

# ==========================================
# 4. 训练逻辑
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 4096

def train():
    dataset = MultiModeRadarDataset1D('./radar_datasets', seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 初始化 1D 模型
    model = Unet1D(dim=64, dim_mults=(1, 2, 4, 8), channels=8, self_condition=True).to(DEVICE)
    # model = Transformer1D(
    #     dim = 128,          # Embedding 维度
    #     depth = 6,          # Transformer 层数
    #     heads = 8,          # 多头注意力头数
    #     dim_head = 32,      # 每个头的维度
    #     channels = 8,       # 4(数据)+4(自条件)
    #     self_condition = True
    # ).to(DEVICE)
    diffusion = GaussianDiffusion1D(
        model, 
        seq_length=SEQ_LEN, 
        timesteps=1000, 
        objective='pred_v', 
        beta_schedule='cosine', 
        auto_normalize=False
    ).to(DEVICE)

    optimizer = Adam(diffusion.parameters(), lr=1e-4)
    
    print("开始 1D 多通道训练...")
    step = 0
    loss_history = []
    
    with tqdm(total=50000) as pbar:
        while step <= 50000:
            for clean, corrupted in dataloader:
                clean, corrupted = clean.to(DEVICE), corrupted.to(DEVICE)
                
                # 训练步
                loss = diffusion(clean, x_self_cond=corrupted)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                loss_history.append(loss.item())
                pbar.set_description(f'Loss: {loss.item():.6f}')
                pbar.update(1)
                
                # 每 5000 步进行一次 5 数据集联合评估
                if step > 0 and step % 5000 == 0:
                    # 1. 绘制当前的 Loss 曲线
                    plot_loss(loss_history)
                    # 2. 抽取 5 个数据集进行绘图对比
                    evaluate_multi_datasets(model, diffusion, dataset, DEVICE, step)
                    # 3. 保存模型权重
                    torch.save(diffusion.state_dict(), f'repair_1d_model_{step}.pt')
                
                step += 1
                if step >= 50000: break

if __name__ == "__main__":
    train()