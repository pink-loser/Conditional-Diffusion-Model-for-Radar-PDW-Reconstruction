import numpy as np
import pickle
import os

# ==========================================
# 1. 引用你提供的原始函数 (直接使用)
# ==========================================

def generate_modulation_pulses(mode, params):
    if mode == 'fixed':
        return np.array([params['value']] * params['length'], dtype=float)
    elif mode == 'staggered':
        group_params = params['staggered_params']
        length = params['length']
        num_params = len(group_params)
        if num_params == 0: return np.array([])
        full_repeats = length // num_params
        remainder = length % num_params
        result = group_params * full_repeats + group_params[:remainder]
        return np.array(result, dtype=float)
    elif mode == 'group':
        staggered_params = params['group_params']
        target_length = params['length']
        result, current_length = [], 0
        while current_length < target_length:
            for value, duration in staggered_params:
                if current_length >= target_length: break
                repeat = min(duration, target_length - current_length)
                result.extend([value] * repeat)
                current_length += repeat
        return np.array(result, dtype=float)
    elif mode == 'slippery':
        start, end, step, total_length = params['start'], params['end'], params['step'], params['length']
        if step == 0: return np.array([start] * total_length, dtype=float)
        if (step > 0 and start <= end) or (step < 0 and start >= end):
            segment = np.arange(start, end + (1e-9 if step > 0 else -1e-9), step, dtype=float)
        else: return np.array([start] * total_length, dtype=float)
        if len(segment) == 0: segment = np.array([start], dtype=float)
        full_repeats, remainder = total_length // len(segment), total_length % len(segment)
        result = np.tile(segment, full_repeats)
        if remainder > 0: result = np.concatenate([result, segment[:remainder]])
        return result
    elif mode == 'jittered':
        value, length = params['value'], params.get('length', 10)
        value_range = value * params.get('range', 0.15)
        return np.random.uniform(value - value_range, value + value_range, length)
    else: raise ValueError("Unsupported modulation type")

def apply_pulse_errors(pri, rf, pw, missing_rate, false_rate, pri_noise=0.0, rf_noise=0.0, pw_noise=0.0):
    # 内部调用你提供的 apply_missing_pulses 和 apply_false_pulses
    pri_missing, rf_missing, pw_missing, missing_indices, modified_indices = \
        apply_missing_pulses(pri, rf, pw, missing_rate)
    
    pri_final, rf_final, pw_final, false_indices = \
        apply_false_pulses(pri_missing, rf_missing, pw_missing, false_rate, len(pri), exclude_indices=modified_indices)
    
    if pri_noise > 0: pri_final += np.random.normal(0, pri_noise, pri_final.shape) * pri_final
    if rf_noise > 0: rf_final += np.random.normal(0, rf_noise, rf_final.shape) * rf_final
    if pw_noise > 0: pw_final += np.random.normal(0, pw_noise, pw_final.shape) * pw_final
    
    return pri_final, rf_final, pw_final, missing_indices, false_indices

# --- 辅助函数：由于 apply_pulse_errors 依赖，需保留 ---
def apply_missing_pulses(pri, rf, pw, missing_rate):
    n = len(pri)
    if missing_rate <= 0 or n == 0: return pri.copy(), rf.copy(), pw.copy(), np.array([]), np.array([])
    num_missing = int(n * missing_rate)
    missing_indices = np.random.choice(n, num_missing, replace=False)
    missing_mask = np.zeros(n, dtype=bool)
    missing_mask[missing_indices] = True
    processed_pri, processed_rf, processed_pw, modified_indices = [], [], [], []
    prev_valid_idx = -1
    for i in range(n):
        if missing_mask[i]:
            if prev_valid_idx != -1:
                processed_pri[prev_valid_idx] += pri[i]
                modified_indices.append(prev_valid_idx)
        else:
            processed_pri.append(pri[i]); processed_rf.append(rf[i]); processed_pw.append(pw[i])
            prev_valid_idx = len(processed_pri) - 1
    return np.array(processed_pri), np.array(processed_rf), np.array(processed_pw), missing_indices, np.unique(modified_indices)

def apply_false_pulses(pri, rf, pw, false_rate, original_length, exclude_indices=None):
    if exclude_indices is None: exclude_indices = np.array([], dtype=int)
    num_false = int(original_length * false_rate)
    if num_false == 0 or len(pri) == 0: return pri.copy(), rf.copy(), pw.copy(), np.array([])
    available_indices = np.setdiff1d(np.arange(len(pri)), exclude_indices)
    if len(available_indices) == 0: return pri.copy(), rf.copy(), pw.copy(), np.array([])
    num_false = min(num_false, len(available_indices))
    false_indices = sorted(np.random.choice(available_indices, num_false, replace=False), reverse=True)
    pri_l, rf_l, pw_l = pri.tolist(), rf.tolist(), pw.tolist()
    for idx in false_indices:
        p_val, w_val = pri_l[idx], pw_l[idx]
        pri_l[idx] = p_val / 2; pri_l.insert(idx + 1, p_val / 2)
        pw_l[idx] = w_val / 2; pw_l.insert(idx + 1, w_val / 2)
        rf_l.insert(idx + 1, rf_l[idx])
    return np.array(pri_l), np.array(rf_l), np.array(pw_l), np.array(false_indices)

# ==========================================
# 2. 批量数据集生成逻辑
# ==========================================

def produce_dataset_by_mode(mode, config, num_samples, save_dir):
    """
    生成特定模式的数据集并保存
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dataset = []
    
    for i in range(num_samples):
        # 1. 生成干净数据 (依据传入的实验配置)
        clean_pri = generate_modulation_pulses(config['pri_mode'], config['pri_params'])
        clean_rf = generate_modulation_pulses(config['rf_mode'], config['rf_params'])
        clean_pw = generate_modulation_pulses(config['pw_mode'], config['pw_params'])
        
        # 2. 随机设置漏脉冲和假脉冲率 (作为随机变量)
        m_rate = np.random.uniform(0.1, 0.4)
        f_rate = np.random.uniform(0.05, 0.2)
        
        # 3. 产生受损数据
        p_err, r_err, w_err, miss_idx, false_idx = apply_pulse_errors(
            clean_pri, clean_rf, clean_pw, 
            missing_rate=m_rate, 
            false_rate=f_rate,
            pri_noise=0.01, rf_noise=0.005, pw_noise=0.01
        )
        
        # 4. 存储原始物理量 (不含掩码，不含归一化)
        sample = {
            'clean': {'pri': clean_pri, 'rf': clean_rf, 'pw': clean_pw},
            'corrupted': {'pri': p_err, 'rf': r_err, 'pw': w_err},
            'labels': {'missing_indices': miss_idx, 'false_indices': false_idx},
            'meta': {'missing_rate': m_rate, 'false_rate': f_rate}
        }
        dataset.append(sample)
    
    file_name = f"pdw_dataset_{mode}.pkl"
    with open(os.path.join(save_dir, file_name), 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Successfully generated {num_samples} samples for [{mode}] -> {file_name}")

# ==========================================
# 3. 使用脚本：按照实验参数生成 5 种模式
# ==========================================

if __name__ == "__main__":
    NUM_SAMPLES = 1000  # 每个模式生成的样本数
    SEQ_LEN = 200      # 依据你之前的实验长度
    SAVE_PATH = "./radar_datasets"

    # 定义五种模式的实验参数配置
    # configs = {
    #     'fixed': {
    #         'pri_params': {'value': 10, 'length': SEQ_LEN}, # 缩小PRI
    #         'rf_mode': 'fixed', 'rf_params': {'value': 10000, 'length': SEQ_LEN},
    #         'pw_mode': 'fixed', 'pw_params': {'value': 20, 'length': SEQ_LEN}
    #     },
    #     'group': {
    #         'pri_params': {'group_params': [(15, 3), (20, 2), (25, 2)], 'length': SEQ_LEN}, # 缩小PRI
    #         'rf_mode': 'fixed', 'rf_params': {'value': 9000, 'length': SEQ_LEN},
    #         'pw_mode': 'fixed', 'pw_params': {'value': 10, 'length': SEQ_LEN}
    #     },
    #     'staggered': {
    #         'pri_params': {'staggered_params': [20, 25, 30], 'length': SEQ_LEN}, # 缩小PRI
    #         'rf_mode': 'fixed', 'rf_params': {'value': 8000, 'length': SEQ_LEN},
    #         'pw_mode': 'jittered', 'pw_params': {'value': 50, 'length': SEQ_LEN}
    #     },
    #     'slippery': {
    #         'pri_params': {'start': 20, 'end': 50, 'step': 2, 'length': SEQ_LEN}, # 缩小PRI
    #         'rf_mode': 'fixed', 'rf_params': {'value': 9000, 'length': SEQ_LEN},
    #         'pw_mode': 'jittered', 'pw_params': {'value': 20, 'length': SEQ_LEN}
    #     },
    #     'jittered': {
    #         'pri_params': {'value': 15, 'length': SEQ_LEN},
    #         'rf_mode': 'fixed', 'rf_params': {'value': 10000, 'length': SEQ_LEN},
    #         'pw_mode': 'fixed', 'pw_params': {'value': 10, 'length': SEQ_LEN}
    #     }
    # }

    # configs = {
    #     # 1. 经典固定搜索模式 (Fixed Search)
    #     # 模拟远程搜索雷达，参数极其稳定，使用大脉宽保证能量
    #     'fixed_search': {
    #         'pri_mode': 'fixed',
    #         'pri_params': {'value': 80, 'length': SEQ_LEN}, 
    #         'rf_mode': 'fixed', 
    #         'rf_params':  {'value': 9400, 'length': SEQ_LEN},
    #         'pw_mode': 'fixed', 
    #         'pw_params':  {'value': 10, 'length': SEQ_LEN}
    #     },

    #     # 2. 参差抗盲速模式 (Staggered Anti-Blind)
    #     # 模拟火控雷达搜索阶段，PRI 切换以解距离/速度模糊，PW 随之微调
    #     'staggered_anti_blind': {
    #         'pri_mode': 'staggered',
    #         'pri_params': {'staggered_params': [40, 45, 50, 55], 'length': SEQ_LEN},
    #         'rf_mode': 'fixed', 
    #         'rf_params':  {'value': 9000, 'length': SEQ_LEN},
    #         'pw_mode': 'staggered', 
    #         'pw_params':  {'staggered_params': [2.0, 2.5, 3.0, 3.5], 'length': SEQ_LEN}
    #     },

    #     # 3. 频率/脉宽组变抗干扰 (Group EP)
    #     # 模拟电子保护模式，每组脉冲切换频率和脉宽，增加侦察难度
    #     'group_ep': {
    #         'pri_mode': 'group',
    #         'pri_params': {'group_params': [(30, 10), (40, 10)], 'length': SEQ_LEN},
    #         'rf_mode': 'group', 
    #         'rf_params':  {'group_params': [(9200, 20), (9500, 20)], 'length': SEQ_LEN},
    #         'pw_mode': 'group', 
    #         'pw_params':  {'group_params': [(8, 20), (4, 20)], 'length': SEQ_LEN}
    #     },

    #     # 4. 线性扫描低截获模式 (LPI Slippery)
    #     # 模拟低截获概率雷达，参数连续线性变化，PW 逐渐压缩
    #     'slippery_lpi': {
    #         'pri_mode': 'slippery',
    #         'pri_params': {'start': 60, 'end': 30, 'step': -0.2, 'length': SEQ_LEN},
    #         'rf_mode': 'slippery', 
    #         'rf_params':  {'start': 8500, 'end': 9500, 'step': 5, 'length': SEQ_LEN},
    #         'pw_mode': 'slippery', 
    #         'pw_params':  {'start': 12, 'end': 4, 'step': -0.05, 'length': SEQ_LEN}
    #     }
    # }

    # 重新定义四种具有物理耦合特性的模式
    configs = {
        # 1. 经典固定搜索 (Fixed Search) - 耦合：低重频大脉宽
        # 模拟远程搜索，能量积聚需求导致 PRI 大时 PW 必须足够大
        'fixed_search': {
            'pri_mode': 'fixed',
            'pri_params': {'value': 100, 'length': SEQ_LEN}, 
            'rf_mode': 'fixed', 
            'rf_params':  {'value': 9400, 'length': SEQ_LEN},
            'pw_mode': 'fixed', 
            'pw_params':  {'value': 15, 'length': SEQ_LEN} # 保持长脉冲
        },

        # 2. 参差抗盲速 (Staggered Anti-Blind) - 耦合：固定占空比 (Duty Cycle)
        # 针对 PW 拟合差：让 PW 的参差比例严格对应 PRI，提供强线性相关性
        'staggered_anti_blind': {
            'pri_mode': 'staggered',
            'pri_params': {'staggered_params': [40, 50, 60, 70], 'length': SEQ_LEN},
            'rf_mode': 'fixed', 
            'rf_params':  {'value': 9000, 'length': SEQ_LEN},
            # PW 与 PRI 同步参差，且数值 = PRI * 0.05，模型更容易学习这种倍数关系
            'pw_mode': 'staggered', 
            'pw_params':  {'staggered_params': [2.0, 2.5, 3.0, 3.5], 'length': SEQ_LEN}
        },

        # 3.  dwells 组变模式 (Group Agility) - 耦合：同步跳变
        # 模拟电子保护。关键点：pri, rf, pw 的组长度(duration)必须完全一致
        'group_agility': {
            'pri_mode': 'group',
            'pri_params': {'group_params': [(30, 15), (50, 15)], 'length': SEQ_LEN}, # 每15个脉冲切一次
            'rf_mode': 'group', 
            'rf_params':  {'group_params': [(9200, 15), (9800, 15)], 'length': SEQ_LEN}, # 同步切频率
            'pw_mode': 'group', 
            'pw_params':  {'group_params': [(3, 15), (6, 15)], 'length': SEQ_LEN} # 同步切脉宽
        },

        # 4. 线性扫频模式 (Slippery LPI) - 耦合：线性演变关系
        # 模拟扫频截获。当 PRI 线性缩短时，RF 线性增加，PW 线性压缩
        'slippery_lpi': {
            'pri_mode': 'slippery',
            'pri_params': {'start': 80, 'end': 40, 'step': -0.4, 'length': SEQ_LEN},
            'rf_mode': 'slippery', 
            'rf_params':  {'start': 8500, 'end': 9500, 'step': 10, 'length': SEQ_LEN},
            'pw_mode': 'slippery', 
            'pw_params':  {'start': 10, 'end': 4, 'step': -0.06, 'length': SEQ_LEN}
        }
    }

    # 循环生成
    for mode, config in configs.items():
        produce_dataset_by_mode(mode, config, NUM_SAMPLES, SAVE_PATH)