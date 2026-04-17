# encoding: utf-8
"""
篮球场景特定的数据增强
针对持球人识别的痛点：遮挡、姿态变化、运动模糊
"""

import random
from PIL import Image, ImageDraw


class BasketballAugmentation:
    """篮球场景数据增强"""

    def __init__(self, occlusion_prob=0.25, pose_prob=0.25):
        self.occlusion_prob = occlusion_prob
        self.pose_prob = pose_prob

    def __call__(self, img):
        """应用篮球场景增强"""
        # 随机选择增强类型
        rand_val = random.random()

        if rand_val < self.occlusion_prob:
            # 模拟多人包夹遮挡
            return self.random_occlusion(img)
        elif rand_val < self.occlusion_prob + self.pose_prob:
            # 模拟持球姿态变化
            return self.random_pose_change(img)
        else:
            return img

    def random_occlusion(self, img):
        """模拟多人包夹遮挡 - 在图像上添加随机矩形遮挡"""
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # 随机生成1-2个遮挡区域（模拟其他球员）
        num_occlusions = random.randint(1, 2)
        for _ in range(num_occlusions):
            # 遮挡区域大小（占图像的10-30%）
            occ_w = int(width * random.uniform(0.1, 0.3))
            occ_h = int(height * random.uniform(0.1, 0.3))

            # 随机位置
            x1 = random.randint(0, width - occ_w)
            y1 = random.randint(0, height - occ_h)
            x2 = x1 + occ_w
            y2 = y1 + occ_h

            # 使用随机颜色（模拟球衣颜色）
            color = tuple(random.randint(50, 200) for _ in range(3))
            draw.rectangle([x1, y1, x2, y2], fill=color)

        return img

    def random_pose_change(self, img):
        """模拟持球姿态变化 - 水平翻转增强"""
        # 水平翻转是最简单的姿态变化模拟
        # 可以模拟左右手运球、左右侧身等姿态
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
