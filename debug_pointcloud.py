import numpy as np

def debug_pointcloud_data(pts, rgb):
    """诊断点云数据的问题"""
    print("=== 点云数据诊断 ===")
    
    # 1. 检查数据是否存在
    if pts is None:
        print("❌ 错误：pts 是 None")
        return False
    if rgb is None:
        print("❌ 错误：rgb 是 None") 
        return False
    
    print("✅ pts 和 rgb 都不是 None")
    
    # 2. 检查数据类型
    print(f"pts 类型: {type(pts)}")
    print(f"rgb 类型: {type(rgb)}")
    
    if not isinstance(pts, np.ndarray):
        print("❌ 错误：pts 不是 numpy 数组")
        return False
    if not isinstance(rgb, np.ndarray):
        print("❌ 错误：rgb 不是 numpy 数组")
        return False
    
    print("✅ pts 和 rgb 都是 numpy 数组")
    
    # 3. 检查数据形状
    print(f"pts 形状: {pts.shape}")
    print(f"rgb 形状: {rgb.shape}")
    
    if len(pts.shape) != 2 or pts.shape[1] != 3:
        print(f"❌ 错误：pts 形状应该是 (N, 3)，实际是 {pts.shape}")
        return False
    if len(rgb.shape) != 2 or rgb.shape[1] != 3:
        print(f"❌ 错误：rgb 形状应该是 (N, 3)，实际是 {rgb.shape}")
        return False
    if pts.shape[0] != rgb.shape[0]:
        print(f"❌ 错误：pts 和 rgb 的点数不匹配：{pts.shape[0]} vs {rgb.shape[0]}")
        return False
    
    print(f"✅ 数据形状正确，共 {pts.shape[0]} 个点")
    
    # 4. 检查数据范围
    print(f"pts 范围: X[{pts[:, 0].min():.3f}, {pts[:, 0].max():.3f}] "
          f"Y[{pts[:, 1].min():.3f}, {pts[:, 1].max():.3f}] "
          f"Z[{pts[:, 2].min():.3f}, {pts[:, 2].max():.3f}]")
    
    print(f"rgb 范围: R[{rgb[:, 0].min():.3f}, {rgb[:, 0].max():.3f}] "
          f"G[{rgb[:, 1].min():.3f}, {rgb[:, 1].max():.3f}] "
          f"B[{rgb[:, 2].min():.3f}, {rgb[:, 2].max():.3f}]")
    
    # 5. 检查RGB值范围
    if rgb.min() < 0 or rgb.max() > 1:
        print("⚠️  警告：RGB值应该在[0,1]范围内")
        if rgb.max() > 1:
            print("   建议：如果RGB值在[0,255]范围，请除以255")
            print("   rgb = rgb / 255.0")
    else:
        print("✅ RGB值在正确范围[0,1]内")
    
    # 6. 检查是否有无效值
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        print("❌ 错误：pts 包含 NaN 或 Inf 值")
        return False
    if np.any(np.isnan(rgb)) or np.any(np.isinf(rgb)):
        print("❌ 错误：rgb 包含 NaN 或 Inf 值")
        return False
    
    print("✅ 数据中没有 NaN 或 Inf 值")
    
    # 7. 检查点云是否太大或太小
    pts_range = np.max(pts, axis=0) - np.min(pts, axis=0)
    pts_scale = np.max(pts_range)
    print(f"点云尺度: {pts_scale:.6f}")
    
    if pts_scale < 1e-6:
        print("⚠️  警告：点云尺度太小，可能看不见")
    elif pts_scale > 1e6:
        print("⚠️  警告：点云尺度太大，可能超出视野")
    else:
        print("✅ 点云尺度合理")
    
    # 8. 统计信息
    print(f"点云中心: ({np.mean(pts, axis=0)})")
    print(f"颜色统计: 平均值({np.mean(rgb, axis=0)}) 标准差({np.std(rgb, axis=0)})")
    
    return True

def suggest_fixes(pts, rgb):
    """建议修复方案"""
    print("\n=== 修复建议 ===")
    
    # RGB值范围修正
    if rgb.max() > 1:
        print("1. RGB值范围修正:")
        print("   rgb = rgb / 255.0  # 如果原来是[0,255]范围")
        print("   或")
        print("   rgb = np.clip(rgb, 0, 1)  # 强制限制在[0,1]范围")
    
    # 点云尺度调整
    pts_scale = np.max(np.max(pts, axis=0) - np.min(pts, axis=0))
    if pts_scale < 0.01:
        print("2. 点云可能太小，尝试放大:")
        print("   pts = pts * 100  # 放大100倍")
    elif pts_scale > 100:
        print("2. 点云可能太大，尝试缩小:")
        print("   pts = pts / 100  # 缩小100倍")
    
    # 立方体尺寸调整
    print("3. 调整点的显示尺寸:")
    print("   # 如果点太小看不见")
    print("   point_mesh.apply_transform(sp.Transforms.Scale(0.01))  # 增大10倍")
    print("   # 如果点太大")
    print("   point_mesh.apply_transform(sp.Transforms.Scale(0.001))  # 缩小2倍")
    
    # 数据下采样
    if pts.shape[0] > 50000:
        print("4. 点云太密集，建议下采样:")
        print("   indices = np.random.choice(len(pts), size=10000, replace=False)")
        print("   pts_sampled = pts[indices]")
        print("   rgb_sampled = rgb[indices]")
    
    print("5. 检查相机位置:")
    print("   确保相机能看到点云区域")
    print("   main.camera = sp.Camera(position=[0, 0, 2], look_at=[0, 0, 0])")

# 使用示例
if __name__ == "__main__":
    print("这是一个诊断脚本，用于检查点云数据")
    print("使用方法:")
    print("from debug_pointcloud import debug_pointcloud_data, suggest_fixes")
    print("debug_pointcloud_data(your_pts, your_rgb)")
    print("suggest_fixes(your_pts, your_rgb)")