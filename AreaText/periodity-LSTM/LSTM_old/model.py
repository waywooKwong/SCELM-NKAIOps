import torch
import torch.nn as nn

'''
这段代码本身是一个类的定义，所以单独运行是没有意义的。它需要被实例化为一个对象，然后调用其中的方法才能执行。这段代码的功能看起来是一个用于时间序列预测的模型，其中包含了一些预训练的语言模型（如 GPT-2、BERT 等）来处理输入数据，并进行相应的预测。

要使这段代码运行，你需要做以下几步：

定义一个 configs 对象，包含你所需要的模型参数，例如 task_name、pred_len、seq_len、d_ff、llm_dim、patch_len、stride、llm_model、llm_layers、prompt_domain、content 等。
根据 configs 对象实例化一个 Model 对象，例如 model = Model(configs)。
将你的输入数据作为参数传递给 model 的 forward 方法，例如 output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)，其中 x_enc、x_mark_enc、x_dec、x_mark_dec 是模型输入数据。
如果需要的话，处理 output 来获取模型的预测结果。
'''
'''
输入样例：

包括以下几个部分：
x_enc：编码器的输入序列，包含历史时间步的数据。
x_mark_enc：编码器的输入序列的标记，可能包含额外的信息，比如时间戳等。
x_dec：解码器的输入序列，通常是一个向量，包含待预测的时间步数。
x_mark_dec：解码器的输入序列的标记，同样可能包含额外的信息。
这里给出一个简单的示例，假设输入序列的长度为 10，预测的步数为 5，输入向量的维度为 3。

# 生成示例数据
x_enc = torch.randn(1, 10, 3)  # (batch_size, seq_len, input_dim)
x_mark_enc = torch.randn(1, 10)  # (batch_size, seq_len)
x_dec = torch.randn(1, 5, 3)  # (batch_size, pred_len, input_dim)
x_mark_dec = torch.randn(1, 5)  # (batch_size, pred_len)

# 示例配置
class Configs:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.pred_len = 5
        self.seq_len = 10
        self.d_ff = 512
        self.llm_dim = 768
        self.patch_len = 16
        self.stride = 8
        self.llm_model = 'GPT2'
        self.llm_layers = 12
        self.prompt_domain = True
        self.content = 'Example description.'
        self.dropout = 0.1
        self.enc_in = 512

# 实例化配置
configs = Configs()

# 实例化模型
model = Model(configs)

# 运行模型
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

print("Output shape:", output.shape)  # 打印输出形状

'''

class Model(nn.Module):

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        #将prompt转换为 模型的 token 输入
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags