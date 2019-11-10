import astra


class AstraWrapper:
    def __init__(self, angles):
        self.angles = angles

    def forward(self, x):
        vol_geom = astra.create_vol_geom(x.shape[1], x.shape[2], x.shape[0])
        phantom_id = astra.data3d.create('-vol', vol_geom, data=x)
        proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, x.shape[0], x.shape[1], self.angles)

        return astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)

    def backproject(self, proj_id, s, bs):
        vol_geom = astra.create_vol_geom(s, s, bs)
        rec_id = astra.data3d.create('-vol', vol_geom)

        # Set up the parameters for a reconstruction algorithm using the GPU
        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        return astra.data3d.get(rec_id)

    def forward_single(self, x):
        vol_geom = astra.create_vol_geom(x.shape[0], x.shape[1])
        proj_geom = astra.create_proj_geom('parallel', 1.0, x.shape[0], -self.angles)
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

        return astra.create_sino(x, proj_id)

    def fbp(self, x):
        s = x.shape[0]
        proj_id, _ = self.forward_single(x)
        vol_geom = astra.create_vol_geom(s, s)
        rec_id = astra.data2d.create('-vol', vol_geom)

        # create configuration
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        cfg['option'] = {'FilterType': 'Ram-Lak'}

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        return astra.data2d.get(rec_id)

