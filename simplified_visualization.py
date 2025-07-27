import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用无头后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5):
    """
    生成相机视锥体的顶点
    
    参数:
    - img_size: (W, H) 图像尺寸
    - K: 3x3 内参矩阵
    - W2C: 4x4 世界到相机的变换矩阵
    - frustum_length: 视锥体长度
    """
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # 构建相机视锥体的顶点 (相机坐标系下)
    frustum_points = np.array([
        [0., 0., 0.],                          # 视锥体原点
        [-half_w, -half_h, frustum_length],    # 左上角
        [half_w, -half_h, frustum_length],     # 右上角
        [half_w, half_h, frustum_length],      # 右下角
        [-half_w, half_h, frustum_length]      # 左下角
    ])
    
    # 将视锥体从相机坐标系变换到世界坐标系
    C2W = np.linalg.inv(W2C)
    frustum_points_h = np.hstack((frustum_points, np.ones((frustum_points.shape[0], 1))))
    frustum_points_world = (C2W @ frustum_points_h.T).T
    frustum_points_world = frustum_points_world[:, :3] / frustum_points_world[:, 3:4]
    
    return frustum_points_world


def visualize_with_matplotlib(pts, rgb, W2C_matrices, K_matrices, images=None, frustum_size=0.1):
    """
    使用matplotlib可视化RGB点云和相机位姿
    
    参数:
    - pts: 点云坐标 (N, 3)
    - rgb: 点云颜色 (N, 3)，值范围[0,1]
    - W2C_matrices: 世界到相机变换矩阵列表
    - K_matrices: 内参矩阵列表
    - images: RGB图像列表（可选）
    - frustum_size: 视锥体大小
    """
    # 创建3D图形
    fig = plt.figure(figsize=(15, 10))
    
    # 主要的3D可视化
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 1. 可视化点云
    if pts is not None and rgb is not None:
        # 如果点云太多，进行下采样
        if len(pts) > 5000:
            indices = np.random.choice(len(pts), size=5000, replace=False)
            pts_vis = pts[indices]
            rgb_vis = rgb[indices]
        else:
            pts_vis = pts
            rgb_vis = rgb
            
        ax1.scatter(pts_vis[:, 0], pts_vis[:, 1], pts_vis[:, 2], 
                   c=rgb_vis, s=1, alpha=0.6)
    
    # 2. 可视化相机位姿和视锥体
    img_size = (640, 480)  # 默认图像尺寸
    
    for i, (W2C, K) in enumerate(zip(W2C_matrices, K_matrices)):
        # 获取相机位置
        C2W = np.linalg.inv(W2C)
        cam_pos = C2W[:3, 3]
        
        # 绘制相机位置
        ax1.scatter(cam_pos[0], cam_pos[1], cam_pos[2], 
                   c='red', s=100, marker='^', label=f'Camera {i+1}' if i < 3 else "")
        
        # 获取视锥体顶点
        frustum_points = get_camera_frustum(img_size, K, W2C, frustum_length=frustum_size)
        
        # 绘制视锥体线条
        # 从相机中心到四个角点
        for j in range(1, 5):
            ax1.plot([frustum_points[0, 0], frustum_points[j, 0]], 
                    [frustum_points[0, 1], frustum_points[j, 1]], 
                    [frustum_points[0, 2], frustum_points[j, 2]], 
                    'g-', alpha=0.6, linewidth=1)
        
        # 绘制图像平面的矩形框
        for j in range(1, 5):
            next_j = j + 1 if j < 4 else 1
            ax1.plot([frustum_points[j, 0], frustum_points[next_j, 0]], 
                    [frustum_points[j, 1], frustum_points[next_j, 1]], 
                    [frustum_points[j, 2], frustum_points[next_j, 2]], 
                    'g-', alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D场景：点云 + 相机位姿')
    ax1.legend()
    
    # 设置相等的坐标轴比例
    if pts is not None:
        max_range = np.array([pts[:, 0].max()-pts[:, 0].min(), 
                             pts[:, 1].max()-pts[:, 1].min(), 
                             pts[:, 2].max()-pts[:, 2].min()]).max() / 2.0
        mid_x = (pts[:, 0].max()+pts[:, 0].min()) * 0.5
        mid_y = (pts[:, 1].max()+pts[:, 1].min()) * 0.5
        mid_z = (pts[:, 2].max()+pts[:, 2].min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 3. 显示图像（如果提供）
    if images is not None and len(images) > 0:
        for i, img in enumerate(images[:3]):  # 最多显示3张图像
            ax_img = fig.add_subplot(2, 3, 4 + i)
            ax_img.imshow(img)
            ax_img.set_title(f'相机 {i+1} 图像')
            ax_img.axis('off')
    
    # 4. 相机位置信息表
    ax_table = fig.add_subplot(223)
    ax_table.axis('off')
    
    table_data = []
    table_data.append(['相机ID', 'X坐标', 'Y坐标', 'Z坐标'])
    
    for i, W2C in enumerate(W2C_matrices):
        C2W = np.linalg.inv(W2C)
        cam_pos = C2W[:3, 3]
        table_data.append([f'相机{i+1}', f'{cam_pos[0]:.2f}', f'{cam_pos[1]:.2f}', f'{cam_pos[2]:.2f}'])
    
    table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0], 
                          cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax_table.set_title('相机位置信息', pad=20)
    
    plt.tight_layout()
    plt.savefig('pointcloud_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    print("可视化完成！图像已保存为 'pointcloud_visualization.png'")


def create_example_data():
    """创建示例数据用于测试"""
    print("创建示例数据...")
    
    # 1. 创建示例点云（一个立方体形状的点云）
    n_points = 2000
    
    # 创建一个有结构的点云（立方体 + 一些随机点）
    # 立方体的边界
    cube_points = []
    for i in range(8):
        for j in range(8):
            for k in range(8):
                if i in [0, 7] or j in [0, 7] or k in [0, 7]:  # 只要边框
                    cube_points.append([i/7.0 - 0.5, j/7.0 - 0.5, k/7.0 - 0.5])
    
    cube_points = np.array(cube_points)
    
    # 添加一些随机点
    random_points = np.random.randn(n_points - len(cube_points), 3) * 0.3
    pts = np.vstack([cube_points, random_points])
    
    # 为点云分配颜色（基于位置）
    rgb = np.zeros_like(pts)
    rgb[:, 0] = (pts[:, 0] + 1) / 2  # R通道基于X坐标
    rgb[:, 1] = (pts[:, 1] + 1) / 2  # G通道基于Y坐标
    rgb[:, 2] = (pts[:, 2] + 1) / 2  # B通道基于Z坐标
    rgb = np.clip(rgb, 0, 1)
    
    # 2. 创建示例相机参数
    n_cameras = 4
    W2C_matrices = []
    K_matrices = []
    images = []
    
    for i in range(n_cameras):
        # 创建环绕物体的相机位置
        angle = i * 2 * np.pi / n_cameras
        radius = 2.0
        cam_pos = np.array([np.cos(angle) * radius, np.sin(angle) * radius, 1.0])
        
        # 创建朝向原点的相机
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # 计算相机坐标系
        z_axis = cam_pos - look_at
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # 构建旋转矩阵
        R = np.column_stack([x_axis, y_axis, z_axis])
        t = -R @ cam_pos
        
        # 世界到相机的变换矩阵
        W2C = np.eye(4)
        W2C[:3, :3] = R
        W2C[:3, 3] = t
        W2C_matrices.append(W2C)
        
        # 内参矩阵
        K = np.array([[500, 0, 320],
                      [0, 500, 240],
                      [0, 0, 1]], dtype=np.float32)
        K_matrices.append(K)
        
        # 创建示例图像（渐变图像）
        img = np.zeros((480, 640, 3))
        for y in range(480):
            for x in range(640):
                img[y, x, 0] = x / 640.0  # R通道
                img[y, x, 1] = y / 480.0  # G通道
                img[y, x, 2] = (x + y) / (640.0 + 480.0)  # B通道
        
        # 添加一些图案使图像更有识别性
        center_x, center_y = 320, 240
        for y in range(480):
            for x in range(640):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 50:
                    img[y, x] = [1, 1, 1]  # 白色圆形
                elif 50 <= dist < 60:
                    img[y, x] = [0, 0, 0]  # 黑色边框
        
        images.append(img)
    
    return pts, rgb, W2C_matrices, K_matrices, images


def print_data_info(pts, rgb, W2C_matrices, K_matrices, images):
    """打印数据的基本信息"""
    print("\n=== 数据信息 ===")
    print(f"点云数量: {len(pts)} 个点")
    print(f"点云坐标范围:")
    print(f"  X: [{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]")
    print(f"  Y: [{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]")
    print(f"  Z: [{pts[:, 2].min():.2f}, {pts[:, 2].max():.2f}]")
    
    print(f"\n相机数量: {len(W2C_matrices)}")
    for i, W2C in enumerate(W2C_matrices):
        C2W = np.linalg.inv(W2C)
        cam_pos = C2W[:3, 3]
        print(f"  相机{i+1}位置: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})")
    
    if images:
        print(f"\n图像数量: {len(images)}")
        print(f"图像尺寸: {images[0].shape}")
    
    print("\n=== 可视化功能说明 ===")
    print("1. 3D散点图显示RGB点云")
    print("2. 红色三角形表示相机位置")
    print("3. 绿色线条表示相机视锥体")
    print("4. 下方显示相机拍摄的图像")
    print("5. 表格显示相机位置坐标")


if __name__ == "__main__":
    print("=== RGB点云和图像3D可视化示例 ===\n")
    
    # 创建示例数据
    pts, rgb, W2C_matrices, K_matrices, images = create_example_data()
    
    # 打印数据信息
    print_data_info(pts, rgb, W2C_matrices, K_matrices, images)
    
    # 进行可视化
    print("\n开始可视化...")
    visualize_with_matplotlib(
        pts=pts,
        rgb=rgb, 
        W2C_matrices=W2C_matrices,
        K_matrices=K_matrices,
        images=images,
        frustum_size=0.3
    )
    
    print("\n=== 如何使用你自己的数据 ===")
    print("1. 准备点云数据: pts (N, 3), rgb (N, 3)")
    print("2. 准备相机参数: W2C矩阵列表, K矩阵列表")
    print("3. 准备图像数据: images列表 (可选)")
    print("4. 调用 visualize_with_matplotlib() 函数")
    print("\n示例:")
    print("visualize_with_matplotlib(my_pts, my_rgb, my_W2C, my_K, my_images)")