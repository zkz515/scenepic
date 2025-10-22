import numpy as np
import scenepic as sp
from skimage.transform import downscale_local_mean


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    """
    生成相机视锥体的顶点、线条和颜色
    
    参数:
    - img_size: (W, H) 图像尺寸
    - K: 3x3 内参矩阵
    - W2C: 4x4 世界到相机的变换矩阵
    - frustum_length: 视锥体长度
    - color: 视锥体颜色
    """
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # 构建相机视锥体的顶点 (相机坐标系下)
    frustum_points = np.array([[0., 0., 0.],                          # 视锥体原点
                               [-half_w, -half_h, frustum_length],    # 左上角
                               [half_w, -half_h, frustum_length],     # 右上角
                               [half_w, half_h, frustum_length],      # 右下角
                               [-half_w, half_h, frustum_length]])    # 左下角
    
    # 定义视锥体的线条连接
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # 将视锥体从相机坐标系变换到世界坐标系
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]
    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    """将多个视锥体合并为一个线条集合"""
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 每个视锥体5个顶点
    merged_lines = np.zeros((N*8, 2))       # 每个视锥体8条线
    merged_colors = np.zeros((N*8, 3))      # 每条线都有颜色

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    return merged_points, merged_lines, merged_colors


def vis_pointcloud(scene, pts, rgb):
    """
    可视化RGB点云
    
    参数:
    - scene: scenepic场景对象
    - pts: 点云坐标 (N, 3)
    - rgb: 点云颜色 (N, 3)，值范围[0,1]
    """
    mesh = scene.create_mesh("pointcloud")
    mesh.shared_color = np.array([0.7, 0.7, 0.7])  # 默认颜色
    
    # 使用小立方体表示每个点
    mesh.add_cube()
    mesh.apply_transform(sp.Transforms.Scale(0.002))  # 缩放到很小
    mesh.enable_instancing(positions=pts, colors=rgb)
    return mesh


def vis_cameras(scene, W2C_matrices, K_matrices, frustum_size=0.5):
    """
    可视化相机位姿（视锥体）
    
    参数:
    - scene: scenepic场景对象
    - W2C_matrices: 世界到相机变换矩阵列表
    - K_matrices: 内参矩阵列表
    - frustum_size: 视锥体大小
    """
    frustums = []
    img_size = (640, 480)  # 默认图像尺寸，你可以根据实际情况修改
    
    for W2C, K in zip(W2C_matrices, K_matrices):
        frustum = get_camera_frustum(img_size, K, W2C, frustum_length=frustum_size)
        frustums.append(frustum)

    points, lines, colors = frustums2lineset(frustums)
    lines = lines.astype(np.int64)

    mesh = scene.create_mesh("cameras", shared_color=sp.Color(0.0, 1.0, 0.0))
    mesh.add_lines(
        start_points=points[lines[:, 0], :],
        end_points=points[lines[:, 1], :]
    )
    return mesh


def vis_images(scene, W2C_matrices, K_matrices, images, frustum_size=0.5):
    """
    可视化RGB图像投影到3D空间
    
    参数:
    - scene: scenepic场景对象
    - W2C_matrices: 世界到相机变换矩阵列表
    - K_matrices: 内参矩阵列表
    - images: RGB图像列表，每个图像形状为(H, W, 3)
    - frustum_size: 图像平面距离相机的距离
    """
    img_meshes = []
    
    for i, (W2C, K, img) in enumerate(zip(W2C_matrices, K_matrices, images)):
        # 创建纹理
        texture_id = f"frame_{i:02d}"
        texture = scene.create_image(image_id=texture_id)
        
        # 处理图像数据
        if img.max() > 1:  # 如果是0-255范围，归一化到0-1
            img_data = img.astype(np.float32) / 255.0
        else:
            img_data = img.astype(np.float32)
        
        # 可选：降采样以提高性能
        img_data_scaled = downscale_local_mean(img_data, (2, 2, 1))
        texture.from_numpy(img_data_scaled)
        
        # 获取视锥体几何
        h, w = img.shape[:2]
        img_size = (w, h)
        frustum_points, _, _ = get_camera_frustum(img_size, K, W2C, frustum_length=frustum_size)
        
        # 图像四个角点对应视锥体的后四个点
        frustum_image_points = frustum_points[1:, :]
        
        # 创建纹理网格
        img_mesh = scene.create_mesh(f"image_{i}", texture_id=texture_id)
        img_mesh.double_sided = True
        img_mesh.add_mesh_without_normals(
            frustum_image_points,
            np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32),  # 两个三角形构成矩形
            uvs=np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)  # UV坐标
        )
        img_meshes.append(img_mesh)
    
    return img_meshes


def visualize_pointcloud_and_images(pts, rgb, W2C_matrices, K_matrices, images, 
                                   frustum_size=0.01, filename='visualization.html'):
    """
    主要的可视化函数：将RGB点云和RGB图像一起可视化
    
    参数:
    - pts: 点云坐标 (N, 3)
    - rgb: 点云颜色 (N, 3)，值范围[0,1]
    - W2C_matrices: 世界到相机变换矩阵列表，每个矩阵形状(4, 4)
    - K_matrices: 内参矩阵列表，每个矩阵形状(3, 3)
    - images: RGB图像列表，每个图像形状(H, W, 3)
    - frustum_size: 相机视锥体和图像平面的大小
    - filename: 输出HTML文件名
    """
    # 创建场景
    scene = sp.Scene()
    
    # 创建所有网格对象
    all_meshes = []
    
    # 1. 可视化点云
    if pts is not None and rgb is not None:
        point_mesh = vis_pointcloud(scene, pts, rgb)
        all_meshes.append(point_mesh)
    
    # 2. 可视化图像
    if images is not None and len(images) > 0:
        img_meshes = vis_images(scene, W2C_matrices, K_matrices, images, frustum_size)
        all_meshes.extend(img_meshes)
    
    # 3. 可视化相机位姿
    cam_mesh = vis_cameras(scene, W2C_matrices, K_matrices, frustum_size)
    all_meshes.append(cam_mesh)
    
    # 创建3D画布
    main = scene.create_canvas_3d(width=1200, height=800,
                                  shading=sp.Shading(bg_color=sp.Colors.White))
    
    # 设置默认相机位置
    main.camera = sp.Camera(position=[0, 0, 2], look_at=[0, 0, 0])
    
    # 创建帧并添加所有网格
    frame = main.create_frame(meshes=all_meshes)
    
    # 保存为HTML文件
    scene.save_as_html(filename, title="RGB点云和图像可视化")
    print(f"可视化结果已保存到: {filename}")


# 示例用法
def create_example_data():
    """创建示例数据用于测试"""
    # 1. 创建示例点云（一个简单的立方体点云）
    n_points = 1000
    pts = np.random.randn(n_points, 3) * 0.5  # 随机点
    rgb = np.random.rand(n_points, 3)  # 随机颜色
    
    # 2. 创建示例相机参数
    n_cameras = 3
    W2C_matrices = []
    K_matrices = []
    images = []
    
    for i in range(n_cameras):
        # 创建环绕的相机位置
        angle = i * 2 * np.pi / n_cameras
        cam_pos = np.array([np.cos(angle) * 2, np.sin(angle) * 2, 1])
        
        # 创建世界到相机的变换矩阵
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # 简化的相机变换（这里只是示例，实际应用中需要正确的变换矩阵）
        W2C = np.eye(4)
        W2C[:3, 3] = -cam_pos
        W2C_matrices.append(W2C)
        
        # 内参矩阵
        K = np.array([[500, 0, 320],
                      [0, 500, 240],
                      [0, 0, 1]], dtype=np.float32)
        K_matrices.append(K)
        
        # 创建示例图像
        img = np.random.rand(480, 640, 3)  # 随机图像
        images.append(img)
    
    return pts, rgb, W2C_matrices, K_matrices, images


if __name__ == "__main__":
    # 创建示例数据
    pts, rgb, W2C_matrices, K_matrices, images = create_example_data()
    
    # 进行可视化
    visualize_pointcloud_and_images(
        pts=pts,
        rgb=rgb, 
        W2C_matrices=W2C_matrices,
        K_matrices=K_matrices,
        images=images,
        frustum_size=0.1,
        filename='pointcloud_images_visualization.html'
    )