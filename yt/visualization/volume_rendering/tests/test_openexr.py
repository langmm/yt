# REQUIREMENTS: mamba install OpenEXR
# export CFLAGS="-I/Users/langmm/mambaforge/envs/yt/include/OpenEXR
#   -I/Users/langmm/mambaforge/envs/yt/include/Imath"
# pip install OpenEXR
# TODO:
#   - Where should this live? Data objects? As a frontend? viz?
#   - Should the C OpenEXR be a dependency? Python?
#   - How should requirements be added? As an option like the backends?
#   - specified in volume_render as fname or separate field?
import os
import shutil
import tempfile
import pytest

import numpy as np

import yt
from yt.testing import requires_file, fake_random_ds  # , requires_module
from yt.visualization.volume_rendering.api import (
    BoxSource,
    LineSource,
    Scene,
    create_volume_source,
)


ISOGAL = "IsolatedGalaxy/galaxy0030/galaxy0030"


# @requires_module("openexr")
class TestOpenEXR:
    # This toggles using a temporary directory. Turn off to examine images.
    use_tmpdir = True

    @pytest.fixture
    def OpenEXR(self):
        try:
            from yt.utilities.lib.openexr import exr_tools as OpenEXR
        except ImportError:
            pytest.skip("OpenEXR not installed")
        return OpenEXR

    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        np.random.seed(0)
        if self.use_tmpdir:
            self.curdir = os.getcwd()
            # Perform I/O in safe place instead of yt main dir
            self.tmpdir = tempfile.mkdtemp()
            os.chdir(self.tmpdir)
        else:
            self.curdir, self.tmpdir = None, None
        yield
        if self.use_tmpdir:
            os.chdir(self.curdir)
            shutil.rmtree(self.tmpdir)

    @requires_file(ISOGAL)
    def test_isogal(self, OpenEXR):
        fname = "isogal.exr"
        ds = yt.load(ISOGAL)
        im, sc = yt.volume_render(ds, field=("gas", "density"),
                                  fname=fname)
        assert os.path.isfile(fname)
        # TODO: Test InputFile
        # o = OpenEXR.InputFile(fname)
        # for k in 'RGBAZ':
        #     assert f"{k}.source_00" in o.channels
        # h = o.header()
        # print(h)
        # TODO: Check size and data?

    @pytest.mark.parametrize('composite,scale', [
        ('layer', False),
        ('deep', False),
        ('flatten', False),
        ('layer', True),
        ('deep', True),
        ('flatten', True),
    ])
    def test_opaque(self, composite, scale, OpenEXR):
        suffix = f"_{composite}"
        if scale:
            suffix += "_scaled"
        fname = f"opaque{suffix}.exr"
        ds = fake_random_ds(64)
        dd = ds.sphere(ds.domain_center, 0.45 * ds.domain_width[0])
        ds.field_info[ds.field_list[0]].take_log = False

        sc = Scene()
        cam = sc.add_camera(ds)
        cam.resolution = (512, 512)
        vr = create_volume_source(dd, field=ds.field_list[0])
        vr.transfer_function.clear()
        vr.transfer_function.grey_opacity = True
        vr.transfer_function.map_to_colormap(0.0, 1.0, scale=3.0,
                                             colormap="Reds")
        sc.add_source(vr, keyname="sphere")

        cam.set_width(1.8 * ds.domain_width)
        cam.lens.setup_box_properties(cam)

        # DRAW SOME LINES
        npoints = 100
        vertices = np.random.random([npoints, 2, 3])
        colors = np.random.random([npoints, 4])
        colors[:, 3] = 0.10

        box_source = BoxSource(
            ds.domain_left_edge, ds.domain_right_edge,
            color=[1.0, 1.0, 1.0, 1.0]
        )
        sc.add_source(box_source, keyname="opaque_box")

        LE = (ds.domain_left_edge + np.array([0.1, 0.0, 0.3])
              * ds.domain_left_edge.uq)
        RE = (ds.domain_right_edge - np.array([0.1, 0.2, 0.3])
              * ds.domain_left_edge.uq)
        color = np.array([0.0, 1.0, 0.0, 0.10])
        box_source = BoxSource(LE, RE, color=color)
        sc.add_source(box_source, keyname="transparent_box")

        line_source = LineSource(vertices, colors)
        sc.add_source(line_source, keyname="transparent_lines")
        sc.save_exr(fname=fname, composite=composite, scale=scale)
