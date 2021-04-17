from nose.tools import assert_equal
from torch_radon import Volume2D, Volume3D


def test_volume_2d():
    volume = Volume2D(64)

    assert_equal(volume.cfg.width, 64)
    assert_equal(volume.cfg.height, 64)
    assert_equal(volume.cfg.depth, 0)
    assert_equal(volume.cfg.is_3d, False)
    assert_equal(volume.cfg.dx, 0.0)
    assert_equal(volume.cfg.dy, 0.0)
    assert_equal(volume.cfg.dz, 0.0)

    volume = Volume2D(64, 128, (1.0, 2.0))
    assert_equal(volume.cfg.width, 128)
    assert_equal(volume.cfg.height, 64)
    assert_equal(volume.cfg.depth, 0)
    assert_equal(volume.cfg.is_3d, False)
    assert_equal(volume.cfg.dx, 1.0)
    assert_equal(volume.cfg.dy, 2.0)
    assert_equal(volume.cfg.dz, 0.0)


def test_volume_3d():
    volume = Volume3D(64)
    assert_equal(volume.cfg.width, 64)
    assert_equal(volume.cfg.height, 64)
    assert_equal(volume.cfg.depth, 64)
    assert_equal(volume.cfg.is_3d, True)
    assert_equal(volume.cfg.dx, 0.0)
    assert_equal(volume.cfg.dy, 0.0)
    assert_equal(volume.cfg.dz, 0.0)

    volume = Volume3D(64, 128)
    assert_equal(volume.cfg.width, 128)
    assert_equal(volume.cfg.height, 128)
    assert_equal(volume.cfg.depth, 64)
    assert_equal(volume.cfg.is_3d, True)
    assert_equal(volume.cfg.dx, 0.0)
    assert_equal(volume.cfg.dy, 0.0)
    assert_equal(volume.cfg.dz, 0.0)

    volume = Volume3D(64, 128, 256, (1.0, 2.0, 3.0))
    assert_equal(volume.cfg.width, 256)
    assert_equal(volume.cfg.height, 128)
    assert_equal(volume.cfg.depth, 64)
    assert_equal(volume.cfg.is_3d, True)
    assert_equal(volume.cfg.dx, 1.0)
    assert_equal(volume.cfg.dy, 2.0)
    assert_equal(volume.cfg.dz, 3.0)