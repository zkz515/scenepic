import scenepic as sp
import numpy as np
import http.server
import socketserver

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

def start_http_server(path, PORT):
    DIRECTORY = str(path)
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DIRECTORY, **kwargs)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()

def vis_pointcloud(pts, rgb):
    """visualize the point cloud with the color"""
    scene = sp.Scene()
    mesh = scene.create_mesh("mesh")
    mesh.shared_color = np.array([0.7, 0.7, 0.7])
    
    mesh.add_cube()
    mesh.apply_transform(sp.Transforms.Scale(0.002)) 
    mesh.enable_instancing(positions=pts, colors=rgb)
    
    main = scene.create_canvas_3d(width=1600, height=1600,
                                  shading=sp.Shading(bg_color=sp.Colors.White))
    main.camera = default_camera()
    
    # ✅ 修正：将mesh包装成列表
    frame1 = main.create_frame(meshes=[mesh])  # 关键修改：用方括号包装
    
    filename = 'pc.html'
    scene.save_as_html(filename, title='test_bi')
    start_http_server(".", 8097)

# 使用示例
# vis_pointcloud(final_points, final_colors)