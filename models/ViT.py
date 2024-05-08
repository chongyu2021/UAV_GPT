from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch


class OWL:
    def __init__(self, model_path):
        """
        初始化OWL类的实例

        参数：
        - model_path (str): 模型路径，用于载入OWL-ViT模型

        返回：
        无返回值
        """
        self.processor = OwlViTProcessor.from_pretrained(model_path)  # 载入OWL-ViT模型的处理器
        self.model = OwlViTForObjectDetection.from_pretrained(model_path)  # 载入OWL-ViT目标检测模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行设备为GPU或CPU
        self.model.to(self.device)  # 将模型移动到指定设备上

    def detect_objects(self, image, text, threshold=0.1):
        """
        使用OWL-ViT模型进行目标检测

        参数：
        - image (PIL.Image.Image): 输入图像，用于目标检测
        - text (str): 输入文本，可能包含与目标检测相关的信息
        - threshold (float, optional): 目标检测阈值，范围在0到1之间，默认为0.1

        返回：
        - detected_boxes (list): 检测到的边界框列表，每个边界框由四个整数坐标值构成，格式为[x_min, y_min, x_max, y_max]
        """
        try:
            inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)  # 将图像和文本转换为模型输入格式，并移动到指定设备上
            outputs = self.model(**inputs)  # 使用模型进行目标检测

            target_sizes = torch.Tensor([image.size[::-1]])  # 获取目标尺寸信息
            results = self.processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)  # 后处理目标检测结果

            boxes = results[0]["boxes"]  # 提取检测到的边界框
            detected_boxes = [[int(i) for i in box.tolist()] for box in boxes]  # 将边界框转换为列表格式

            return detected_boxes  # 返回检测到的边界框列表
        except Exception as e:
            print(f"Error during object detection: {e}")  # 打印异常信息
            return []  # 返回空列表表示未检测到任何边界框
