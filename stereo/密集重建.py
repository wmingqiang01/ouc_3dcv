import open3d as o3d

# 创建一个简单的点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

# 显示点云
o3d.visualization.draw_geometries([pcd], window_name="Test Point Cloud", width=800, height=600)
