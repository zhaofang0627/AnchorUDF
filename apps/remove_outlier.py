import argparse
import trimesh
import open3d as o3d


def convert_to_ply(path):
    m = trimesh.load(path)
    m.export(path+'.ply')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run outlier removing'
    )
    parser.add_argument('--file_path', type=str, help='path to file')
    parser.add_argument('--nb_neighbors', type=int, default=5,
                        help='how many neighbors are taken into account to calculate the average distance for a point.')
    parser.add_argument('--std_ratio', type=float, default=10.0,
                        help='threshold based on the standard deviation of the average distances across the point cloud.')

    args = parser.parse_args()

    file_path = args.file_path
    nb_neighbors = args.nb_neighbors
    std_ratio = args.std_ratio

    convert_to_ply(file_path)

    pc = o3d.io.read_point_cloud(file_path+'.ply')
    cl, _ = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    o3d.io.write_point_cloud(file_path+'.ply', cl)

    print('Done! The result is saved at ' + file_path + '.ply')
