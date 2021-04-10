from torch_radon_cuda import VolumeCfg


class Volume2D:
    def __init__(self, height, width=-1, center=(0.0, 0.0), voxel_size=(1.0, 1.0)):
        if width <= 0:
            width = height

        self.height = height
        self.width = width

        self.center = center
        self.voxel_size = voxel_size

        self.cfg = self.to_cfg()

    def to_cfg(self) -> VolumeCfg:
        return VolumeCfg(
            0, self.height, self.width,
            0.0, self.center[1], self.center[0],
            1.0, self.voxel_size[1], self.voxel_size[0],
            False
        )

    def num_dimensions(self):
        return 2


class Volume3D:
    def __init__(self, depth, height, width, center=(0.0, 0.0, 0.0), voxel_size=(1.0, 1.0, 1.0)):
        if height <= 0:
            height = depth

        if width <= 0:
            width = height

        self.depth = depth
        self.height = height
        self.width = width

        self.center = center
        self.voxel_size = voxel_size

        self.cfg = self.to_cfg()

    def to_cfg(self) -> VolumeCfg:
        return VolumeCfg(
            self.depth, self.height, self.width,
            self.center[2], self.center[1], self.center[0],
            self.voxel_size[2], self.voxel_size[1], self.voxel_size[0],
            True
        )

    def num_dimensions(self):
        return 3
