from models.ViT import OWL
from models.BLIP import BLIP
from envs.airsim_env import AirsimEnv
import numpy as np

owl_vit = OWL(model_path="D:\Models\OWL-ViT")
blip = BLIP(model_path="D:\Models\BLIP")
env = AirsimEnv()


def takeoff():
    """
    起飞操作

    参数：
    无

    返回：
    无返回值
    """
    env.takeoff()
    print("起飞成功！")


def land():
    """
    降落操作

    参数：
    无

    返回：
    无返回值
    """
    env.land()
    print("降落成功！")

def get_position():
    """
    获取当前无人机的位置信息。

    参数：
    无

    返回：
    - position (dict): 包含当前位置信息的字典，格式为{"x": x坐标, "y": y坐标, "z": z坐标}
    """
    position = env.get_position()  # 调用环境对象的get_position函数获取无人机位置信息
    print("当前位置是",position)
    return position  # 返回位置信息字典

def move_to_position(x, y, z):
    """
    将飞行器移动到指定位置

    参数：
    - x (float): 目标位置的x坐标
    - y (float): 目标位置的y坐标
    - z (float): 目标位置的z坐标

    返回：
    无返回值
    """
    env.move_to_position(x, y, z)
    print("成功移动到", f"({x},{y},{z})")


def rotate_to_angle(raw):
    """
    将飞行器旋转到指定角度

    参数：
    - raw (float): 目标角度的yaw值，单位为度（0-360）

    返回：
    无返回值
    """
    env.rotate_to_angle(raw)
    print("成功转向", f"({raw}度)")


def get_distance(box):
    """
    检测到目标区域的距离

    参数：
    - box (list): 检测距离的范围

    返回：
    - distance (float): 到目标区域的最近距离
    """
    depth_array = env.get_depth_image_array()
    x1, y1, x2, y2 = box
    # 提取目标区域内的深度值
    target_depths = depth_array[y1:y2, x1:x2]
    # 计算目标区域的距离（这里使用最大值表示）
    distance = np.min(target_depths)
    print("目标区域距当前位置", distance,"米")
    return distance


def search(keyword):
    """
    在前置摄像头中搜索包含指定关键字的物体

    参数：
    - cameras (list): 待搜索的相机列表
    - keyword (str): 搜索关键字

    返回：
    - results (list): 包含搜索结果的列表，每个元素为[x1,y1,x2,y2]
    """
    image = env.get_scene_image()
    boxes = owl_vit.detect_objects(image, keyword)
    print("搜索到结果：",boxes)
    return boxes









