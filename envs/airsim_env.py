import airsim
from PIL import Image
from io import BytesIO

import numpy as np


class AirsimEnv:
    def __init__(self):
        """
        初始化Airsim环境

        参数：
        无

        返回：
        无返回值
        """
        self.client = airsim.MultirotorClient()  # 创建Airsim客户端对象
        self.client.confirmConnection()  # 确认连接到Airsim模拟器

    def takeoff(self):
        """
        启动Airsim环境，使飞行器起飞

        参数：
        无

        返回：
        无返回值
        """
        self.client.enableApiControl(True)  # 启用API控制
        self.client.armDisarm(True)  # 解锁飞行器
        self.client.takeoffAsync().join()  # 异步起飞，并等待完成

    def land(self):
        """
        使飞行器降落

        参数：
        无

        返回：
        无返回值
        """
        self.client.landAsync().join()  # 异步使飞行器降落，并等待完成
        self.client.enableApiControl(False)  # 关闭API控制
        self.client.armDisarm(False)  # 锁飞行器

    def move_to_position(self, x, y, z):
        """
        将飞行器移动到指定位置

        参数：
        - x (float): 目标位置的x坐标
        - y (float): 目标位置的y坐标
        - z (float): 目标位置的z坐标

        返回：
        无返回值
        """
        self.client.moveToPositionAsync(x, y, z, 1).join()  # 异步移动飞行器到指定位置，并等待完成

    def rotate_to_angle(self, yaw):
        """
        将飞行器旋转到指定角度

        参数：
        - yaw (float): 目标角度的yaw值，单位为度（0-360）

        返回：
        无返回值
        """
        self.client.rotateToYawAsync(yaw).join()  # 异步旋转飞行器到指定角度，并等待完成

    def get_position(self):
        """
        获取当前位置信息

        参数：
        无

        返回：
        - position (dict): 包含当前位置信息的字典，格式为{"x": x坐标, "y": y坐标, "z": z坐标}
        """
        kinematics = self.client.simGetGroundTruthKinematics()  # 获取飞行器当前位置信息
        position = {"x": kinematics.position.x_val, "y": kinematics.position.y_val,
                    "z": kinematics.position.z_val}  # 提取位置信息并构建字典
        return position  # 返回位置信息字典

    def get_scene_image(self, camera_name='front_center'):
        """
        获取当前相机图像

        参数：
        - camera_name (str): 相机名称，默认为'front_center'

        返回：
        - image (PIL.Image.Image): 当前相机图像
        """
        responses = self.client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene)])  # 获取相机图像
        image = Image.open(BytesIO(responses[0].image_data_uint8))  # 将图像数据转换为PIL图像对象
        return image  # 返回相机图像

    def get_depth_image(self, camera_name='front_center'):
        """
        获取当前相机的深度图像

        参数：
        - camera_name (str): 相机名称，默认为'front_center'

        返回：
        - depth_image (PIL.Image.Image): 当前相机的深度图像
        """
        responses = self.client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective)])  # 获取深度图像
        depth_image = Image.open(BytesIO(responses[0].image_data_uint8))  # 将图像数据转换为PIL图像对象
        return depth_image  # 返回深度图像

    def get_depth_image_array(self, camera_name='front_center', depth_threshold=100):
        """
        获取深度图像数组

        参数:
        - self: 实例对象
        - camera_name (str): 相机名称，默认为'front_center'
        - depth_threshold (int): 深度阈值，默认为100

        返回:
        - img_depth_array (numpy.ndarray): 深度图像数组，表示深度图像数据

        """
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)])
        img_depth_array = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
        return img_depth_array  # 返回深度数组
