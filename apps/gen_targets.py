import numpy as np
import os
import trimesh
import argparse
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans


def generate_targets(subject):
    out_path = os.path.join(root_path, 'TARGET', subject)
    anchor_path = os.path.join(root_path, 'KEY_POINT', subject)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if not os.path.exists(anchor_path):
        os.makedirs(anchor_path)

    out_file = os.path.join(out_path, 'boundary_{}_direct_samples.npz'.format(sigma))
    anchor_file = os.path.join(anchor_path, 'key_point_{}.npz'.format(point_num))

    if os.path.exists(out_file):
        return

    # generate target
    mesh = trimesh.load(os.path.join(OBJ_path, subject, '%s.obj' % (subject+'.obj_scaled')))
    surface_points, _ = trimesh.sample.sample_surface(mesh, sample_num)
    sample_points = surface_points + np.random.normal(scale=sigma, size=surface_points.shape)

    np.random.shuffle(sample_points)

    sample_point_count = 10000000
    surface_point_cloud = mesh.sample(sample_point_count, return_index=False)
    kd_tree = KDTree(surface_point_cloud)
    distances, ind = kd_tree.query(sample_points)
    distances = distances.astype(np.float32).reshape(-1)
    neighbors = surface_point_cloud[ind.reshape(-1)]

    # generate anchor
    surface_points, _ = trimesh.sample.sample_surface(mesh, 100000)
    kmeans = KMeans(n_clusters=point_num, random_state=0).fit(surface_points)
    key_points = kmeans.cluster_centers_

    np.savez(out_file, points=sample_points, df=distances, neighbors=neighbors)
    np.savez(anchor_file, kp=key_points)
    print('Finished {}'.format(out_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run target generating'
    )
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--point_num', type=float)

    args = parser.parse_args()

    root_path = args.dataroot
    OBJ_path = os.path.join(root_path, 'GEO', 'OBJ')

    subjects = os.listdir(OBJ_path)

    sigma = args.sigma  # 0.08, 0.02, 0.003
    point_num = args.point_num  # 600

    if sigma == 0.08:
        sample_num = 200000
    else:
        sample_num = 1000000

    i = 0
    for s in subjects:
        generate_targets(s)
        i += 1
        print(i)
