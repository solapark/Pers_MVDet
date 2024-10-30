import sys
sys.path.append('/home/sapark/ped/MVDet')
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from kornia.geometry.transform import warp_perspective
from torchvision.models.vgg import vgg11, vgg16
from multiview_detector.models.resnet import resnet18

import matplotlib.pyplot as plt


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18', wh_train=None):
        super().__init__()
        self.wh_train = wh_train
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])

        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))

        #imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
        #                                                                   dataset.base.extrinsic_matrices,
        #                                                                   dataset.base.worldgrid2worldcoord_mat)
        self.intrinsic_matrices = dataset.base.intrinsic_matrices
        self.worldgrid2worldcoord_mat = dataset.base.worldgrid2worldcoord_mat
        # img
        self.img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        self.img_zoom_mat = np.diag(np.append(self.img_reduce, [1]))
        # map
        self.map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        #self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
        #                  for cam in range(self.num_cam)]
        self.reudced_img_coord_map = self.create_img_coord_map(self.upsample_shape + [1])

        self.fix_extrinsic_matrices = dataset.fix_extrinsic_matrices
        if self.fix_extrinsic_matrices : 
            self.proj_mats = self.get_proj_mats(dataset.base.extrinsic_matrices)

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:1')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'vgg16':
            base = vgg16().features
            base = list(base.children())

            base[16] = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=2, dilation=2)
            base[23] = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=4, dilation=4)
            base[30] = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=8, dilation=8)
        
            base = nn.Sequential(*base)

            split = 10
            self.base_pt1 = base[:split].to('cuda:1')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to('cuda:1')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        if self.wh_train :
            self.img_wh = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to('cuda:0')

        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to('cuda:0')
        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, padding=1), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')
        pass

    def forward(self, data, visualize=False):
        imgs, extrinsic_matrices = data
        self.proj_mats = self.get_proj_mats(extrinsic_matrices[0]) if not self.fix_extrinsic_matrices else self.proj_mats

        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        imgs_result = []
        for cam in range(self.num_cam):
            img_feature = self.base_pt1(imgs[:, cam].to('cuda:1'))
            img_feature = self.base_pt2(img_feature.to('cuda:0'))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to('cuda:0'))
            if self.wh_train is not None:
                img_wh = self.img_wh(img_feature.to('cuda:0'))
                img_res = torch.cat([img_res, img_wh], 1)

            imgs_result.append(img_res)
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
            world_feature = warp_perspective(img_feature.to('cuda:0'), proj_mat, self.reducedgrid_shape)
            if visualize:
                plt.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
                plt.savefig('img_C%d.png'%(cam))
                #plt.show()
                plt.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
                #plt.gca().invert_yaxis()  # Flip y-axis so that 0 is at the bottom
                plt.savefig('feat_C%d.png'%(cam))
                #plt.show()
                
                img_feature_reprojection = warp_perspective(world_feature, torch.inverse(proj_mat), self.upsample_shape)
                plt.imshow(torch.norm(img_feature_reprojection[0].detach(), dim=0).cpu().numpy())
                plt.savefig('img_feature_reprojection_C%d.png'%(cam))
 
                img_xy = self.reudced_img_coord_map
                img_wh = img_res[:, 2:4]
                plt.imshow(torch.norm(img_xy[0, 0:1].detach(), dim=0).cpu().numpy())
                plt.savefig('X_C%d.png'%(cam))
                plt.imshow(torch.norm(img_xy[0, 1:2].detach(), dim=0).cpu().numpy())
                plt.savefig('Y_C%d.png'%(cam))
                plt.imshow(torch.norm(img_wh[0, 0:1].detach(), dim=0).cpu().numpy())
                plt.savefig('W_C%d.png'%(cam))
                plt.imshow(torch.norm(img_wh[0, 1:2].detach(), dim=0).cpu().numpy())
                plt.savefig('H_C%d.png'%(cam))

                warp_xy = self.warp_xy(cam)
                warp_wh = self.warp_wh(img_wh, cam)
                plt.imshow(torch.norm(warp_xy[0, 0:1].detach(), dim=0).cpu().numpy())
                plt.savefig('warpX_C%d.png'%(cam))
                plt.imshow(torch.norm(warp_xy[0, 1:2].detach(), dim=0).cpu().numpy())
                plt.savefig('warpY_C%d.png'%(cam))
                plt.imshow(torch.norm(warp_wh[0, 0:1].detach(), dim=0).cpu().numpy())
                plt.savefig('warpW_C%d.png'%(cam))
                plt.imshow(torch.norm(warp_wh[0, 1:2].detach(), dim=0).cpu().numpy())
                plt.savefig('warpH_C%d.png'%(cam))

            world_features.append(world_feature.to('cuda:0'))

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        if visualize:
            plt.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
            #plt.gca().invert_yaxis()  # Flip y-axis so that 0 is at the bottom
            plt.savefig('world_feat.png')
            #plt.show()
        map_result = self.map_classifier(world_features.to('cuda:0')) #(325, 450)
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')

        if visualize:
            plt.imshow(torch.norm(map_result[0].detach(), dim=0).cpu().numpy())
            #plt.gca().invert_yaxis()  # Flip y-axis so that 0 is at the bottom
            plt.savefig('map.png')
            #plt.show()

        return map_result, imgs_result

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret

    def create_img_coord_map(self, img_size):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x)
        grid_y = torch.from_numpy(grid_y)
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).float()
        return ret

    def get_proj_mats(self, extrinsic_matrices) :
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(self.intrinsic_matrices,
                                                                           np.array(extrinsic_matrices),
                                                                           self.worldgrid2worldcoord_mat)
        proj_mats = [torch.from_numpy(self.map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ self.img_zoom_mat)
                          for cam in range(self.num_cam)]
        return proj_mats

    def warp_xy(self, cam):
        img = self.reudced_img_coord_map
        proj_mat = self.proj_mats[cam].repeat([len(img), 1, 1]).float().to('cuda:0')
        img_warp = warp_perspective(img.to('cuda:0'), proj_mat, self.reducedgrid_shape, mode='bilinear')
        return img_warp

    def warp_wh(self, img, cam):
        proj_mat = self.proj_mats[cam].repeat([len(img), 1, 1]).float().to('cuda:0')
        img_warp = warp_perspective(img.to('cuda:0'), proj_mat, self.reducedgrid_shape, mode='nearest')
        return img_warp
 

def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.datasets.Messytable import Messytable
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataset = frameDataset(Messytable(os.path.expanduser('~/Data/Messytable')), transform=transform, fix_extrinsic_matrices=False)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    #model = PerspTransDetector(dataset)
    model = PerspTransDetector(dataset, arch='vgg11', wh_train='center')
    #model.load_state_dict(torch.load('/home/sapark/ped/MVDet/logs/messytable_frame/default/2024-10-24_12-43-38/MultiviewDetector.pth'))

    map_res, img_res = model(imgs, visualize=True)

    #plt.gca().invert_yaxis()  # Flip y-axis so that 0 is at the bottom
    plt.savefig('map_gt.png')

    pass


if __name__ == '__main__':
    test()
