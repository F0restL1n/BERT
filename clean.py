import pandas as pd

# 输入和输出文件路径
input_file = "D:/AboutCode/Python/Pycharm/Code/BERT/glue_data/MRPC/test_cleaned.tsv"
output_file = "D:/AboutCode/Python/Pycharm/Code/BERT/glue_data/MRPC/test_final.tsv"

# 加载文件并重命名列
df = pd.read_csv(input_file, sep="\t")

# 统一列名为训练/验证集的格式
expected_columns = ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']

# 如果列名不匹配，添加缺失列或调整列名
if 'index' in df.columns:
    df = df.rename(columns={'index': 'Quality'})  # 重命名列，确保一致
if 'Quality' not in df.columns:
    df['Quality'] = -1  # 如果缺少 Quality 列，添加一个默认值列

# 重新排列列顺序，确保与预期格式一致
df = df[expected_columns]

# 保存修正后的文件
df.to_csv(output_file, sep="\t", index=False)

print(f"Fixed file saved to {output_file}")
