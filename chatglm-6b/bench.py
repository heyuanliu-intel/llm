from transformers import AutoTokenizer, AutoModel

# model_id = "THUDM/chatglm-6b"
model_id = "/root/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/1d240ba371910e9282298d4592532d7f0f3e9f3e/"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).cpu().float()
model = model.eval()
print("asking questions now")
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
print(len(response))
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=[])
# print(response)
