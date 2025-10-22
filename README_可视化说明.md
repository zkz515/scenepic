# RGB点云和图像3D可视化说明

## 概述

这个代码库基于 **ScenePic** 库，用于将RGB点云和RGB图像一起进行3D可视化。ScenePic是微软开发的一个轻量级3D可视化库，可以生成交互式的HTML文件。

## 核心功能

### 1. 代码库结构理解

你提供的代码主要包含以下几个核心功能：

- **点云可视化** (`vis_pointcloud`): 将3D点云以小立方体的形式显示，每个点都有自己的RGB颜色
- **相机位姿可视化** (`vis_cameras`): 显示相机的视锥体(frustum)，表示相机的位置和朝向
- **图像可视化** (`vis_images`): 将2D图像投影到3D空间中的对应位置
- **综合可视化** (`vis_points_images_cameras_new`): 将以上三者结合

### 2. 关键概念

#### 相机视锥体 (Camera Frustum)
- 视锥体是一个截椎体，表示相机的视野范围
- 由相机内参矩阵K和外参矩阵W2C确定
- 包含5个顶点：相机中心+图像四个角点在3D空间的投影

#### 坐标系变换
- **W2C矩阵**: 世界坐标系到相机坐标系的变换矩阵 (4×4)
- **K矩阵**: 相机内参矩阵 (3×3)，包含焦距和主点信息

## 使用方法

### 1. 数据准备

你需要准备以下数据：

```python
# 点云数据
pts = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # 形状: (N, 3)
rgb = np.array([[r1, g1, b1], [r2, g2, b2], ...])  # 形状: (N, 3)，值范围[0,1]

# 相机参数
W2C_matrices = [W2C1, W2C2, ...]  # 每个矩阵形状: (4, 4)
K_matrices = [K1, K2, ...]        # 每个矩阵形状: (3, 3)

# 图像数据
images = [img1, img2, ...]         # 每个图像形状: (H, W, 3)
```

### 2. 简单使用

```python
from visualization_example import visualize_pointcloud_and_images

# 调用主要可视化函数
visualize_pointcloud_and_images(
    pts=pts,                    # 点云坐标
    rgb=rgb,                    # 点云颜色
    W2C_matrices=W2C_matrices,  # 相机外参
    K_matrices=K_matrices,      # 相机内参
    images=images,              # RGB图像列表
    frustum_size=0.1,          # 视锥体大小
    filename='my_visualization.html'  # 输出文件名
)
```

### 3. 分步可视化

如果你想要更精细的控制，可以分别调用各个函数：

```python
import scenepic as sp

# 创建场景
scene = sp.Scene()

# 1. 可视化点云
point_mesh = vis_pointcloud(scene, pts, rgb)

# 2. 可视化相机位姿
cam_mesh = vis_cameras(scene, W2C_matrices, K_matrices, frustum_size=0.1)

# 3. 可视化图像
img_meshes = vis_images(scene, W2C_matrices, K_matrices, images, frustum_size=0.1)

# 组合所有网格
all_meshes = [point_mesh, cam_mesh] + img_meshes

# 创建画布和帧
main = scene.create_canvas_3d(width=1200, height=800)
frame = main.create_frame(meshes=all_meshes)

# 保存
scene.save_as_html('custom_visualization.html')
```

## 参数说明

### visualize_pointcloud_and_images 函数参数

- **pts**: 点云坐标，numpy数组，形状(N, 3)
- **rgb**: 点云颜色，numpy数组，形状(N, 3)，值范围[0,1]
- **W2C_matrices**: 世界到相机的变换矩阵列表
- **K_matrices**: 相机内参矩阵列表  
- **images**: RGB图像列表
- **frustum_size**: 控制相机视锥体和图像平面的大小
- **filename**: 输出的HTML文件名

### 重要提示

1. **颜色范围**: 确保RGB颜色值在[0,1]范围内
2. **矩阵格式**: 
   - W2C矩阵应该是4×4的齐次变换矩阵
   - K矩阵应该是3×3的内参矩阵
3. **图像格式**: 图像应该是numpy数组，形状为(H, W, 3)
4. **坐标系**: 注意你的数据使用的坐标系约定

## 交互操作

生成的HTML文件支持以下交互操作：

- **鼠标拖拽**: 旋转视角
- **Shift+拖拽**: 平移视角  
- **鼠标滚轮**: 缩放
- **'r'键**: 重置视角
- **图层控制**: 可以单独开关不同的可视化元素

## 常见问题

### 1. 如果点云太密集怎么办？
可以在调用前对点云进行下采样：
```python
# 下采样点云
indices = np.random.choice(len(pts), size=min(10000, len(pts)), replace=False)
pts_sampled = pts[indices]
rgb_sampled = rgb[indices]
```

### 2. 如何调整点云点的大小？
修改 `vis_pointcloud` 函数中的缩放参数：
```python
mesh.apply_transform(sp.Transforms.Scale(0.005))  # 增大点的尺寸
```

### 3. 如何调整相机视锥体的大小？
调整 `frustum_size` 参数，或者修改函数中的 `frustum_length` 参数。

### 4. 图像显示不正确？
检查：
- 图像的数据类型和值范围
- 相机内外参矩阵是否正确
- 坐标系约定是否一致

## 扩展功能

你可以基于这个框架添加更多功能：

1. **动画支持**: 使用多帧显示运动轨迹
2. **更多几何形状**: 添加线条、箭头等
3. **不同的点云表示**: 使用球体或其他形状代替立方体
4. **颜色映射**: 根据深度或其他属性进行颜色编码

## 依赖库

确保安装了以下库：
```bash
pip install scenepic numpy scikit-image
```

这个可视化系统特别适用于：
- 3D重建结果展示
- SLAM系统的轨迹和地图可视化  
- 多视角几何的调试
- 点云处理结果的展示