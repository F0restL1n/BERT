import os
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from datasets import load_dataset

# 设置环境变量
BERT_BASE_DIR = os.getenv('BERT_BASE_DIR', "D:/AboutCode/Python/Pycharm/Code/BERT/uncased_L-12_H-768_A-12")
GLUE_DIR = os.getenv('GLUE_DIR', "D:/AboutCode/Python/Pycharm/Code/BERT/glue_data")

data_files = {
    "train": "D:\\AboutCode\\Python\\Pycharm\\Code\\BERT\\glue_data\\MRPC\\train_cleaned.tsv",
    "validation": "D:\\AboutCode\\Python\\Pycharm\\Code\\BERT\\glue_data\\MRPC\\dev_cleaned.tsv",
    "test": "D:\\AboutCode\\Python\\Pycharm\\Code\\BERT\\glue_data\\MRPC\\test_final.tsv"
}

# 加载数据集
dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

# 打印列名，确认标签所在的列名
print("Dataset columns:", dataset['train'].column_names)

# 假设标签列名为 'Quality'，如果列名不同，替换为实际名称
LABEL_COLUMN = 'Quality'

# 确保标签为整数类型
def clean_labels(example):
    example[LABEL_COLUMN] = int(example[LABEL_COLUMN])
    return example

# 应用标签清洗函数
dataset = dataset.map(clean_labels)

# 过滤标签，确保只有 0 和 1
dataset = dataset.filter(lambda example: example[LABEL_COLUMN] in [0, 1])

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained(BERT_BASE_DIR)
model = TFBertForSequenceClassification.from_pretrained(BERT_BASE_DIR, num_labels=2)

# 数据预处理：将文本转换为 BERT 输入格式
def preprocess_function(examples):
    return tokenizer(examples['#1 String'], examples['#2 String'],
                     truncation=True, padding="max_length", max_length=128)

# 对数据集进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 转换为 TensorFlow 数据格式
train_dataset = encoded_dataset['train'].to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols=LABEL_COLUMN,
    shuffle=True,
    batch_size=32
)

val_dataset = encoded_dataset['validation'].to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols=LABEL_COLUMN,
    shuffle=False,
    batch_size=32
)

test_dataset = encoded_dataset['test'].to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols=LABEL_COLUMN,
    shuffle=False,
    batch_size=32
)

# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 编译模型
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

# 开始训练
model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# 评估模型
results = model.evaluate(test_dataset)

# 输出评估结果
print("***** Eval results *****")
for key, value in zip(model.metrics_names, results):
    print(f"  {key} = {value}")
