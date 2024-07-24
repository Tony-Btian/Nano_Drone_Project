import pyvista as pv
import pandas as pd

def read_ply(file_path):
    """
    读取点云文件
    :param file_path: 点云文件的路径
    :return: DataFrame包含点云数据
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    start = 0
    for i, line in enumerate(lines):
        if line.startswith("end_header"):
            start = i + 1
            break

    data = [list(map(float, line.strip().split())) for line in lines[start:]]
    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
    return df

def visualize_ply(file_path):
    """
    加载并可视化点云文件
    :param file_path: 点云文件的路径
    """
    try:
        df = read_ply(file_path)
        print(f"成功加载点云文件：{file_path}")
        
        # 创建点云对象
        cloud = pv.PolyData(df[['x', 'y', 'z']].values)
        
        # 设置点的颜色
        cloud['colors'] = df[['x', 'y', 'z']].values
        
        # 可视化点云
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, scalars='colors', point_size=1, render_points_as_spheres=True)
        plotter.show()
    except Exception as e:
        print(f"加载点云文件失败：{e}")

if __name__ == "__main__":
    # 这里替换为你的点云文件路径
    file_path = "point_cloud.ply"
    visualize_ply(file_path)
