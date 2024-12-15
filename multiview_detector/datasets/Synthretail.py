import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset
import json

intrinsic_camera_matrix_filenames = ['CAM1_intrinsics.xml', 'CAM2_intrinsics.xml', 'CAM3_intrinsics.xml']

class Synthretail(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        self.__name__ = 'Synthretail'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [1000, 1500]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 3, 1880+1861
        # x,y actually means i,j in Wildtrack, which correspond to h,w
        self.indexing = 'xy'
        # i,j for world map 
        self.worldgrid2worldcoord_mat = np.array([[0, .01, -7.5], [.01, 0, -5.0], [0, 0, 1]])
        self.intrinsic_matrices = [self.get_intrinsic_matrix(cam) for cam in range(self.num_cam)]
        self.extrinsic_matrices = self.get_extrinsic_matrix()

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 1500
        grid_y = pos // 1500
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return grid_y*1500 + grid_x 

    def get_worldgrid_from_worldcoord(self, world_coord):
        # datasets default unit: centimeter & origin: (-30,-90)
        coord_x, coord_y = world_coord
        grid_x = (coord_x + 7.5) / .01
        grid_y = (coord_y + 5.0) / .01
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        # datasets default unit: centimeter & origin: (-30,-90)
        grid_x, grid_y = worldgrid
        coord_x = -7.5 + .01 * grid_x
        coord_y = -5.0 + .01 * grid_y
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        return intrinsic_matrix

    def get_extrinsic_matrix(self):
        with open(os.path.join(self.root, 'calibrations', 'extrinsic.json')) as json_file:
            data = json.load(json_file)

        matrix_list = []
        for key in data:
            inner_matrices = []
            for sub_key in data[key]:
                inner_matrices.append(data[key][sub_key])
            matrix_list.append(inner_matrices)

        return np.array(matrix_list)

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 1920 - 1), min(bottom, 1080 - 1)]
        return bbox_by_pos_cam


def test():
    import sys
    sys.path.append('/home/sapark/ped/MVDet')
    from multiview_detector.utils.projection import get_imagecoord_from_worldcoord
    dataset = Wildtrack(os.path.expanduser('~/Data/Wildtrack'), )
    pom = dataset.read_pom()

    foot_3ds = dataset.get_worldcoord_from_pos(np.arange(np.product(dataset.worldgrid_shape)))
    errors = []
    for cam in range(dataset.num_cam):
        projected_foot_2d = get_imagecoord_from_worldcoord(foot_3ds, dataset.intrinsic_matrices[cam],
                                                           dataset.extrinsic_matrices[cam])
        for pos in range(np.product(dataset.worldgrid_shape)):
            bbox = pom[pos][cam]
            foot_3d = dataset.get_worldcoord_from_pos(pos)
            if bbox is None:
                continue
            foot_2d = [(bbox[0] + bbox[2]) / 2, bbox[3]]
            p_foot_2d = projected_foot_2d[:, pos]
            p_foot_2d = np.maximum(p_foot_2d, 0)
            p_foot_2d = np.minimum(p_foot_2d, [1920, 1080])
            errors.append(np.linalg.norm(p_foot_2d - foot_2d))

    print(f'average error in image pixels: {np.average(errors)}')
    pass


if __name__ == '__main__':
    test()
