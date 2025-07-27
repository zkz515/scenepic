import scenepic as sp
import numpy as np
import http.server
import socketserver
import threading
import time

def default_camera():
    r = 2.0
    theta = np.pi / 12
    gamma = np.pi / 4
    return sp.Camera(
        center=np.array([
            -r * np.cos(theta) * np.cos(gamma),
            -r * np.cos(theta) * np.sin(gamma),
             r * np.sin(theta)
        ]),
        up_dir=np.array([0., 0., 1.])
    )

def start_http_server_non_blocking(path, PORT, timeout=30):
    """启动非阻塞的HTTP服务器，指定时间后自动停止"""
    DIRECTORY = str(path)
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def run_server():
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🌐 服务器启动: http://localhost:{PORT}")
            print(f"⏰ 将在 {timeout} 秒后自动停止")
            print("💡 现在请在浏览器中访问上面的链接查看点云")
            
            # 设置超时
            end_time = time.time() + timeout
            httpd.timeout = 1  # 每秒检查一次是否超时
            
            while time.time() < end_time:
                httpd.handle_request()
            
            print("⏹️  服务器已停止")
    
    # 在后台线程中运行服务器
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread

def vis_pointcloud_debug(pts, rgb, point_size=0.005, auto_camera=True):
    """调试版本的点云可视化"""
    
    print("=== 点云可视化调试信息 ===")
    print(f"📊 点数: {len(pts)}")
    print(f"📐 pts shape: {pts.shape}, dtype: {pts.dtype}")
    print(f"🎨 rgb shape: {rgb.shape}, dtype: {rgb.dtype}")
    
    # 检查数据范围
    print(f"📍 坐标范围:")
    print(f"   X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"   Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]") 
    print(f"   Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
    
    print(f"🌈 颜色范围:")
    print(f"   R: [{rgb[:,0].min():.3f}, {rgb[:,0].max():.3f}]")
    print(f"   G: [{rgb[:,1].min():.3f}, {rgb[:,1].max():.3f}]")
    print(f"   B: [{rgb[:,2].min():.3f}, {rgb[:,2].max():.3f}]")
    
    # 数据预处理
    pts = pts.astype(np.float32)
    rgb = rgb.astype(np.float32)
    
    # 如果RGB值超出[0,1]范围，进行归一化
    if rgb.max() > 1.0:
        print("🔧 RGB值超出[0,1]范围，进行归一化")
        rgb = rgb / rgb.max()
    
    # 创建场景
    scene = sp.Scene()
    mesh = scene.create_mesh("pointcloud")
    mesh.shared_color = np.array([1.0, 0.0, 0.0])  # 红色作为备用颜色
    
    # 添加立方体几何
    mesh.add_cube()
    
    # 应用缩放变换 - 使用更大的点尺寸
    mesh.apply_transform(sp.Transforms.Scale(point_size))
    print(f"🔍 点尺寸设置为: {point_size}")
    
    # 启用实例化渲染
    mesh.enable_instancing(positions=pts, colors=rgb)
    print("✅ 实例化渲染已启用")
    
    # 创建画布
    main = scene.create_canvas_3d(width=1600, height=1600,
                                  shading=sp.Shading(bg_color=sp.Colors.White))
    
    # 设置相机
    if auto_camera:
        # 根据点云自动设置相机位置
        pts_center = np.mean(pts, axis=0)
        pts_range = np.max(pts, axis=0) - np.min(pts, axis=0)
        max_range = np.max(pts_range)
        
        camera_distance = max_range * 3  # 距离设为点云尺寸的3倍
        camera_pos = pts_center + np.array([camera_distance, camera_distance, camera_distance])
        
        main.camera = sp.Camera(
            center=camera_pos,
            look_at=pts_center,
            up_dir=np.array([0., 0., 1.])
        )
        print(f"📹 自动相机设置:")
        print(f"   中心: {pts_center}")
        print(f"   位置: {camera_pos}")
        print(f"   距离: {camera_distance:.3f}")
    else:
        main.camera = default_camera()
        print("📹 使用默认相机设置")
    
    # 创建帧 - 修正：使用列表
    frame1 = main.create_frame(meshes=[mesh])
    print("🎬 帧创建成功")
    
    # 保存HTML文件
    filename = 'pointcloud_debug.html'
    scene.save_as_html(filename, title='调试版点云可视化')
    print(f"💾 HTML文件已保存: {filename}")
    
    # 启动非阻塞服务器
    server_thread = start_http_server_non_blocking(".", 8097, timeout=60)
    
    return server_thread

def vis_pointcloud_simple(pts, rgb):
    """简化版本 - 只保存HTML，不启动服务器"""
    print("=== 简化版点云可视化 ===")
    
    # 数据预处理
    pts = pts.astype(np.float32)
    rgb = rgb.astype(np.float32)
    
    if rgb.max() > 1.0:
        rgb = rgb / rgb.max()
    
    # 创建场景
    scene = sp.Scene()
    mesh = scene.create_mesh("pointcloud")
    mesh.shared_color = np.array([0.7, 0.7, 0.7])
    
    mesh.add_cube()
    mesh.apply_transform(sp.Transforms.Scale(0.01))  # 更大的点尺寸
    mesh.enable_instancing(positions=pts, colors=rgb)
    
    # 创建画布和相机
    main = scene.create_canvas_3d(width=1600, height=1600,
                                  shading=sp.Shading(bg_color=sp.Colors.White))
    
    # 自动设置相机
    pts_center = np.mean(pts, axis=0)
    pts_range = np.max(pts, axis=0) - np.min(pts, axis=0)
    max_range = np.max(pts_range)
    camera_distance = max_range * 3
    
    main.camera = sp.Camera(
        center=pts_center + np.array([camera_distance, camera_distance, camera_distance]),
        look_at=pts_center,
        up_dir=np.array([0., 0., 1.])
    )
    
    # 创建帧
    frame1 = main.create_frame(meshes=[mesh])
    
    # 保存文件
    filename = 'pointcloud.html'
    scene.save_as_html(filename, title='点云可视化')
    print(f"✅ 文件已保存: {filename}")
    print(f"💡 请手动打开浏览器访问文件: {filename}")
    
    return filename

# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    n_points = 1000
    test_pts = np.random.randn(n_points, 3) * 0.5
    test_rgb = np.random.rand(n_points, 3)
    
    print("测试1: 调试版本（自动启动服务器）")
    vis_pointcloud_debug(test_pts, test_rgb)
    
    print("\n测试2: 简化版本（只保存文件）")
    vis_pointcloud_simple(test_pts, test_rgb)