import scenepic as sp
import numpy as np

def debug_colors(rgb):
    """调试颜色数据"""
    print("=== 颜色数据调试 ===")
    print(f"RGB shape: {rgb.shape}")
    print(f"RGB dtype: {rgb.dtype}")
    print(f"RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"RGB mean: {rgb.mean(axis=0)}")
    print(f"RGB std: {rgb.std(axis=0)}")
    
    # 检查是否有无效值
    if np.any(np.isnan(rgb)):
        print("❌ 警告: RGB中包含NaN值")
    if np.any(np.isinf(rgb)):
        print("❌ 警告: RGB中包含Inf值")
    
    # 检查颜色分布
    unique_colors = len(np.unique(rgb.reshape(-1, rgb.shape[-1]), axis=0))
    print(f"唯一颜色数: {unique_colors}")
    
    return rgb

def vis_pointcloud_color_test(pts, rgb, test_mode="auto"):
    """
    测试不同的颜色设置方法
    
    test_mode: 
    - "auto": 自动检测最佳方法
    - "shared": 只用shared_color
    - "instance": 只用instance colors
    - "both": 同时使用两种颜色
    """
    
    print(f"=== 颜色测试模式: {test_mode} ===")
    
    # 调试颜色数据
    rgb = debug_colors(rgb)
    
    # 确保数据类型正确
    pts = pts.astype(np.float32)
    rgb = rgb.astype(np.float32)
    
    # 归一化RGB到[0,1]范围
    if rgb.max() > 1.0:
        print("🔧 归一化RGB值到[0,1]范围")
        rgb = rgb / rgb.max()
    
    scene = sp.Scene()
    
    if test_mode == "shared":
        # 测试1: 只用shared_color，不用instance colors
        mesh = scene.create_mesh("mesh")
        mesh.shared_color = np.array([1.0, 0.0, 0.0])  # 红色
        mesh.add_cube()
        mesh.apply_transform(sp.Transforms.Scale(0.01))
        mesh.enable_instancing(positions=pts)  # 不传colors参数
        
    elif test_mode == "instance":
        # 测试2: 只用instance colors
        mesh = scene.create_mesh("mesh")
        mesh.shared_color = np.array([0.5, 0.5, 0.5])  # 灰色基础
        mesh.add_cube()
        mesh.apply_transform(sp.Transforms.Scale(0.01))
        mesh.enable_instancing(positions=pts, colors=rgb)
        
    elif test_mode == "both":
        # 测试3: 同时使用
        mesh = scene.create_mesh("mesh")
        mesh.shared_color = np.array([0.0, 1.0, 0.0])  # 绿色基础
        mesh.add_cube()
        mesh.apply_transform(sp.Transforms.Scale(0.01))
        mesh.enable_instancing(positions=pts, colors=rgb)
        
    else:  # auto
        # 自动模式：根据颜色数据决定
        mesh = scene.create_mesh("mesh")
        
        # 检查颜色的变化程度
        color_variance = np.var(rgb, axis=0).sum()
        print(f"颜色方差: {color_variance:.6f}")
        
        if color_variance < 1e-6:
            # 颜色几乎不变，使用平均色作为shared_color
            avg_color = np.mean(rgb, axis=0)
            mesh.shared_color = avg_color
            mesh.add_cube()
            mesh.apply_transform(sp.Transforms.Scale(0.01))
            mesh.enable_instancing(positions=pts)
            print("🎨 使用平均颜色作为shared_color")
        else:
            # 颜色变化较大，使用instance colors
            mesh.shared_color = np.array([0.7, 0.7, 0.7])  # 中性基础色
            mesh.add_cube()
            mesh.apply_transform(sp.Transforms.Scale(0.01))
            mesh.enable_instancing(positions=pts, colors=rgb)
            print("🎨 使用instance colors")
    
    # 设置画布和相机
    main = scene.create_canvas_3d(width=1200, height=1200,
                                  shading=sp.Shading(bg_color=sp.Colors.White))
    
    # 自动设置相机
    pts_center = np.mean(pts, axis=0)
    pts_range = np.max(pts, axis=0) - np.min(pts, axis=0)
    max_range = np.max(pts_range)
    camera_distance = max_range * 2.5
    
    main.camera = sp.Camera(
        center=pts_center + np.array([camera_distance, camera_distance, camera_distance]),
        look_at=pts_center,
        up_dir=np.array([0., 0., 1.])
    )
    
    print(f"📹 相机设置: 中心{pts_center}, 距离{camera_distance:.3f}")
    
    # 创建帧
    frame1 = main.create_frame(meshes=[mesh])
    
    # 保存文件
    filename = f'pointcloud_{test_mode}.html'
    scene.save_as_html(filename, title=f'点云测试_{test_mode}')
    print(f"✅ 保存完成: {filename}")
    
    return filename

def create_test_pointcloud():
    """创建一个有明显颜色变化的测试点云"""
    n_points = 1000
    
    # 创建一个彩色的球形点云
    phi = np.random.uniform(0, 2*np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    u = np.random.uniform(0, 1, n_points)
    
    theta = np.arccos(costheta)
    r = 0.5 * np.cbrt(u)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    pts = np.column_stack([x, y, z])
    
    # 根据位置创建彩色
    rgb = np.zeros_like(pts)
    rgb[:, 0] = (x + 0.5)  # R基于X
    rgb[:, 1] = (y + 0.5)  # G基于Y  
    rgb[:, 2] = (z + 0.5)  # B基于Z
    rgb = np.clip(rgb, 0, 1)
    
    return pts, rgb

def run_all_tests(pts, rgb):
    """运行所有测试模式"""
    test_modes = ["shared", "instance", "both", "auto"]
    
    print("🧪 开始运行所有颜色测试...")
    
    for mode in test_modes:
        try:
            filename = vis_pointcloud_color_test(pts, rgb, mode)
            print(f"✅ {mode} 模式成功: {filename}")
        except Exception as e:
            print(f"❌ {mode} 模式失败: {e}")
        print("-" * 50)

if __name__ == "__main__":
    # 创建测试数据
    print("创建测试点云...")
    test_pts, test_rgb = create_test_pointcloud()
    
    # 运行所有测试
    run_all_tests(test_pts, test_rgb)