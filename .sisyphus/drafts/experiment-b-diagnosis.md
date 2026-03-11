# Draft: 实验B - 对比模型输出诊断

## 目标
对比两种加载方式的模型输出，找出数值差异

## 诊断脚本内容

请将以下脚本保存为 `BenchMARL/diagnose_loading.py` 并运行：

```python
#!/usr/bin/env python3
"""
诊断脚本：对比两种加载方式的模型输出差异
"""

import torch
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.environments import LayupTask
from benchmarl.algorithms import MappoConfig, EnsembleAlgorithmConfig
from benchmarl.models import SequenceModelConfig, EnsembleModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.debug_utils import setup_model_logging
import os
import glob

# 禁用调试日志
setup_model_logging(log_to_console=False, log_to_file=False)

# Dummy callback
class WinRateReportDebounced(Callback):
    def __init__(self, *args, **kwargs): pass
    def on_setup(self): pass
    def on_batch_collected(self, batch): pass

def find_latest_checkpoint(pattern="outputs/**/checkpoints/*.pt"):
    files = glob.glob(pattern, recursive=True)
    return max(files, key=os.path.getmtime) if files else None

def load_via_reload(checkpoint_path):
    """方式1: reload_from_file"""
    print("\n" + "="*60)
    print("方式1: reload_from_file")
    print("="*60)
    exp = Experiment.reload_from_file(checkpoint_path)
    print(f"✓ use_amp: {exp.config.use_amp}, amp_dtype: {exp.amp_dtype}")
    print(f"  total_frames: {exp.total_frames}, n_iters: {exp.n_iters_performed}")
    for group, scaler in exp.grad_scalers.items():
        print(f"  {group} GradScaler scale: {scaler.get_scale():.2e}")
    return exp

def load_via_cont(checkpoint_path):
    """方式2: cont 模式"""
    print("\n" + "="*60)
    print("方式2: cont 模式")
    print("="*60)
    
    os.environ['VMAS_INITIAL_SHOT_THRESHOLD'] = '0.2'
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.restore_file = checkpoint_path
    
    exp = Experiment(
        task=LayupTask.LAYUP.get_from_yaml(),
        algorithm_config=EnsembleAlgorithmConfig({
            "attacker": MappoConfig.get_from_yaml(),
            "defender": MappoConfig.get_from_yaml()
        }),
        model_config=EnsembleModelConfig({
            "attacker": SequenceModelConfig([
                AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_attacker.yaml"),
                GruConfig.get_from_yaml()
            ], [128]),
            "defender": SequenceModelConfig([
                AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_defender.yaml"),
                GruConfig.get_from_yaml()
            ], [128])
        }),
        critic_model_config=AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml"),
        seed=114514,
        config=experiment_config,
        callbacks=[WinRateReportDebounced()]
    )
    print(f"✓ use_amp: {exp.config.use_amp}, amp_dtype: {exp.amp_dtype}")
    print(f"  total_frames: {exp.total_frames}, n_iters: {exp.n_iters_performed}")
    for group, scaler in exp.grad_scalers.items():
        print(f"  {group} GradScaler scale: {scaler.get_scale():.2e}")
    return exp

def compare_params(exp1, exp2):
    """对比模型参数"""
    print("\n" + "="*60)
    print("对比模型参数")
    print("="*60)
    
    for group in ['attacker', 'defender']:
        print(f"\n=== {group.upper()} ===")
        
        # Actor
        actor1 = dict(exp1.losses[group].actor_network.named_parameters())
        actor2 = dict(exp2.losses[group].actor_network.named_parameters())
        
        max_diff = 0
        for name in actor1:
            if name not in actor2:
                print(f"❌ {name}: 只在 exp1 中存在")
                continue
            diff = (actor1[name] - actor2[name]).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                print(f"⚠️ {name}: diff={diff:.2e}")
        
        if max_diff < 1e-6:
            print(f"✅ Actor 参数完全一致 (max_diff={max_diff:.2e})")
        else:
            print(f"❌ Actor 参数存在差异 (max_diff={max_diff:.2e})")
        
        # Critic
        critic1 = dict(exp1.losses[group].critic_network.named_parameters())
        critic2 = dict(exp2.losses[group].critic_network.named_parameters())
        
        max_diff = 0
        for name in critic1:
            if name not in critic2:
                print(f"❌ {name}: 只在 exp1 中存在")
                continue
            diff = (critic1[name] - critic2[name]).abs().max().item()
            max_diff = max(max_diff, diff)
        
        if max_diff < 1e-6:
            print(f"✅ Critic 参数完全一致 (max_diff={max_diff:.2e})")
        else:
            print(f"❌ Critic 参数存在差异 (max_diff={max_diff:.2e})")
        
        # Optimizer
        for opt_name in exp1.optimizers[group]:
            opt1 = exp1.optimizers[group][opt_name]
            opt2 = exp2.optimizers[group][opt_name]
            
            print(f"\nOptimizer {opt_name}:")
            print(f"  exp1 lr: {opt1.param_groups[0]['lr']}")
            print(f"  exp2 lr: {opt2.param_groups[0]['lr']}")
            
            if len(opt1.state) > 0:
                first_key = list(opt1.state.keys())[0]
                s1 = opt1.state[first_key]
                s2 = opt2.state.get(first_key, {})
                
                if 'exp_avg' in s1 and 'exp_avg' in s2:
                    diff = (s1['exp_avg'] - s2['exp_avg']).abs().max().item()
                    print(f"  exp_avg diff: {diff:.2e}")
                if 'exp_avg_sq' in s1 and 'exp_avg_sq' in s2:
                    diff = (s1['exp_avg_sq'] - s2['exp_avg_sq']).abs().max().item()
                    print(f"  exp_avg_sq diff: {diff:.2e}")

def main():
    checkpoint_path = find_latest_checkpoint()
    print(f"Checkpoint: {checkpoint_path}")
    
    exp1 = load_via_reload(checkpoint_path)
    exp2 = load_via_cont(checkpoint_path)
    compare_params(exp1, exp2)
    
    print("\n" + "="*60)
    print("诊断完成")
    print("="*60)

if __name__ == "__main__":
    main()
```

## 运行方法

```bash
cd /workspaces/robocon2025_marl_devcontainer/BenchMARL
python diagnose_loading.py
```

## 预期输出

1. 两种加载方式的基本信息对比
2. GradScaler 状态对比
3. Actor/Critic 参数对比
4. 优化器状态对比

## 关注点

- 如果参数完全一致，问题可能在其他地方
- 如果参数不一致，说明加载过程中有差异