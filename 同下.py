import os
import re
import csv
import matplotlib.pyplot as plt
import pandas as pd

# 创建输出目录
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 正则表达式匹配关键字段
log_pattern = re.compile(
    r"step=(\d+).*?Train Loss:\s*([0-9.eE+-]+).*?Grad Norm:\s*([0-9.eE+-]+).*?"
    r"Lr:\s*([0-9.eE+-]+).*?Consumed Samples:\s*(\d+), Consumed Video Samples:\s*\d+, Consumed Tokens:\s*([0-9,]+)"
)

# 输入日志文件路径
input_log_file = "Pasted_Text_1751854654824.txt"

# 输出文件路径
output_csv_file = os.path.join(output_dir, "metrics.csv")
loss_curve_file = os.path.join(output_dir, "loss_curve.png")
lr_curve_file = os.path.join(output_dir, "lr_curve.png")
grad_norm_curve_file = os.path.join(output_dir, "grad_norm_curve.png")
tokens_curve_file = os.path.join(output_dir, "tokens_curve.png")
all_metrics_file = os.path.join(output_dir, "all_metrics.png")

# 提取日志并写入 CSV
with open(input_log_file, "r") as f_in, open(output_csv_file, "w", newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["step", "loss", "grad_norm", "lr", "consumed_samples", "consumed_tokens"])

    for line in f_in:
        match = log_pattern.search(line)
        if match:
            step, loss, grad_norm, lr, samples, tokens = match.groups()
            tokens = int(tokens.replace(',', ''))  # 去除逗号转为整数
            writer.writerow([step, loss, grad_norm, lr, samples, tokens])

# 加载 CSV 数据
df = pd.read_csv(output_csv_file)
df['step'] = df['step'].astype(int)
df['loss'] = df['loss'].astype(float)
df['grad_norm'] = df['grad_norm'].astype(float)
df['lr'] = df['lr'].astype(float)
df['consumed_samples'] = df['consumed_samples'].astype(int)
df['consumed_tokens'] = df['consumed_tokens'].astype(int)

print("✅ 提取完成，前几行数据如下：")
print(df.head())

# 绘图 1: Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(df['step'], df['loss'], label='Training Loss', color='blue')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(loss_curve_file)
plt.close()

# 绘图 2: Learning Rate Curve
plt.figure(figsize=(10, 5))
plt.plot(df['step'], df['lr'], label='Learning Rate', color='green')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(lr_curve_file)
plt.close()

# 绘图 3: Grad Norm Curve
plt.figure(figsize=(10, 5))
plt.plot(df['step'], df['grad_norm'], label='Gradient Norm', color='orange')
plt.xlabel('Step')
plt.ylabel('Grad Norm')
plt.title('Gradient Norm Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(grad_norm_curve_file)
plt.close()

# 绘图 4: Token 数量增长曲线
plt.figure(figsize=(10, 5))
plt.plot(df['step'], df['consumed_tokens'], label='Consumed Tokens', color='purple')
plt.xlabel('Step')
plt.ylabel('Tokens')
plt.title('Total Tokens Processed')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(tokens_curve_file)
plt.close()

# 绘图 5: 合并所有指标在一个图中
plt.figure(figsize=(14, 10))

plt.subplot(3, 2, 1)
plt.plot(df['step'], df['loss'], label='Loss', color='blue')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(df['step'], df['lr'], label='LR', color='green')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(df['step'], df['grad_norm'], label='Grad Norm', color='orange')
plt.xlabel('Step')
plt.ylabel('Grad Norm')
plt.title('Gradient Norm')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(df['step'], df['consumed_tokens'], label='Tokens', color='purple')
plt.xlabel('Step')
plt.ylabel('Tokens')
plt.title('Consumed Tokens')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(df['step'], df['consumed_samples'], label='Samples', color='brown')
plt.xlabel('Step')
plt.ylabel('Samples')
plt.title('Consumed Samples')
plt.grid(True)

plt.tight_layout()
plt.savefig(all_metrics_file)
plt.close()

print(f"📊 图表已全部保存至 {output_dir}/ 目录下。")