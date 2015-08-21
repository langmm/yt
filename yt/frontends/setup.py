#!/usr/bin/env python
import setuptools


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('frontends', parent_package, top_path)
    config.make_config_py()  # installs __config__.py
    #config.make_svn_version_py()
    config.add_subpackage("art")
    config.add_subpackage("artio")
    config.add_subpackage("athena")
    config.add_subpackage("boxlib")
    config.add_subpackage("chombo")
    config.add_subpackage("eagle")
    config.add_subpackage("enzo")
    config.add_subpackage("fits")
    config.add_subpackage("flash")
    config.add_subpackage("gadget")
    config.add_subpackage("gadget_fof")
    config.add_subpackage("gdf")
    config.add_subpackage("halo_catalog")
    config.add_subpackage("http_stream")
    config.add_subpackage("moab")
    config.add_subpackage("owls")
    config.add_subpackage("owls_subfind")
    config.add_subpackage("ramses")
    config.add_subpackage("rockstar")
    config.add_subpackage("sdf")
    config.add_subpackage("sph")
    config.add_subpackage("stream")
    config.add_subpackage("tipsy")
    config.add_subpackage("ytdata")
    config.add_subpackage("art/tests")
    config.add_subpackage("artio/tests")
    config.add_subpackage("athena/tests")
    config.add_subpackage("boxlib/tests")
    config.add_subpackage("chombo/tests")
    config.add_subpackage("eagle/tests")
    config.add_subpackage("enzo/tests")
    config.add_subpackage("fits/tests")
    config.add_subpackage("flash/tests")
    config.add_subpackage("gadget/tests")
    config.add_subpackage("gadget_fof/tests")
    config.add_subpackage("moab/tests")
    config.add_subpackage("owls/tests")
    config.add_subpackage("owls_subfind/tests")
    config.add_subpackage("ramses/tests")
    config.add_subpackage("rockstar/tests")
    config.add_subpackage("stream/tests")
    config.add_subpackage("tipsy/tests")
    return config
