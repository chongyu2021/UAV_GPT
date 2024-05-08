import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP:
    def __init__(self, model_path):
        """
        初始化BLIP类的实例

        参数：
        - model_path (str): 模型路径，用于载入BLIP模型

        返回：
        无返回值
        """
        self.processor = Blip2Processor.from_pretrained(model_path)  # 载入BLIP模型的处理器
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_path)  # 载入BLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 设置运行设备为GPU或CPU
        self.model.to(self.device)  # 将模型移动到指定设备上

    def generate_text(self, image):
        """
        使用BLIP模型生成文本描述

        参数：
        - image (PIL.Image.Image): 输入图像，用于生成文本描述

        返回：
        - generated_text (str): 生成的文本描述
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)  # 将图像转换为模型输入格式
            generated_ids = self.model.generate(**inputs)  # 使用模型生成文本
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()  # 解码生成的文本

            return generated_text  # 返回生成的文本描述
        except Exception as e:
            print(f"Error during text generation: {e}")  # 打印异常信息
            return None  # 返回空值表示文本生成失败
