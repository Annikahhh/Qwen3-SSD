from qwen_asr import Qwen3ASRModel
model_path = "Qwen/Qwen3-ASR-0.6B"
# 載入模型 (不需要放資料，只要看結構)
asr_wrapper = Qwen3ASRModel.from_pretrained(model_path)
model = asr_wrapper.model
# 印出所有模組名稱
for name, module in model.named_modules():
    print(name)