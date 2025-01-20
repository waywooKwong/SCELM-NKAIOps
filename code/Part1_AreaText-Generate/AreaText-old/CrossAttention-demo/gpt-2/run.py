from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 加载GPT-2模型和分词器
model_name = '/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/gpt-2/GPT2-down'  # 或者 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# # 定义输入文本
# input_text = "could you introduce openAI?"
# 读取.txt文件中的内容
with open('/home/sunyongqian/liuheng/Time-LLM/weihua/CrossAttention-demo/gpt-2/output_read_folder.txt', 'r', encoding='utf-8') as file:
    input_text = file.read()

# # 编码输入文本并生成回复
# input_ids = tokenizer.encode(input_text, return_tensors='pt')
# output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

# # 解码生成的文本
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# # 打印生成的回复
# print(generated_text)
# 编码输入文本并截断到模型的最大序列长度
input_ids = tokenizer.encode(input_text, return_tensors='pt')
max_length = model.config.n_positions
input_ids = input_ids[:, :max_length]

# 设置attention_mask
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

# 生成回复
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 打印生成的回复
print(generated_text)