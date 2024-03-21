import numpy as np
import copy
try:
    from yt.utilities.lib.openexr import exr_tools as OpenEXR
except ImportError:
    OpenEXR = None


class OpenEXRCanvas:
    r"""Output an image to an OpenEXR file.

    Parameters
    ----------
    fname: string
        The name of the file where the image should be saved.
    shape: tuple
        Width and height of the image in pixels.
    channels: sequence of strings, optional
        Names of fields that should be written out as channels. If
        not provided, R, G, B, A, & Z (depth) will be written.
    layers: sequence of strings, optional
        Names of layers that should be added as sub-channels.
    base_layer: string
        Layer that should be treated as the base layer for the image. If
        not provided and layers is provided, the first layer will be treated
        as the base layer.
    scale: bool, optional
        If True, the image values in each channel will be scaled to
        0-255. If False (the default), the raw channel data will be
        returned.
    sigma_clip: float, optional
        Image values greater than this number times the standard
        deviation plus the mean of the image will be clipped before
        saving. Useful for enhancing images as it gets rid of rare
        high pixel values.
        Default: None

        floor(vals > std_dev*sigma_clip + mean)
    max_nsamples: integer, optional
        Maximum number of samples for each pixel in the image if it
        will be a deep image.
        Default: 1

    """
    def __init__(self, fname, shape, channels=None,
                 layers=None, base_layer=None,
                 scale=False, sigma_clip=None, max_nsamples=1):
        if OpenEXR is None:
            raise RuntimeError("OpenEXR is not installed")
        self.fname = fname
        self.shape = shape
        if channels is None:
            channels = ['R', 'G', 'B', 'A', 'Z']
        self.channels = channels
        if layers is None:
            layers = []
        if base_layer is None and layers:
            base_layer = layers[0]
        self.base_layer = base_layer
        self.layers = layers
        self.scale = scale
        self.sigma_clip = sigma_clip
        self.max_nsamples = max_nsamples
        self.fd = OpenEXR.OutputFile(fname, shape,
                                     {x: np.float64 for
                                      x in self.allChannels},
                                     maxNSamples=max_nsamples)
        if max_nsamples > 1:
            self.fd.setSampleCountAll(max_nsamples)

    @property
    def allChannels(self):
        r"""list: Channel names for all layers in the image"""
        out = copy.deepcopy(self.channels)
        for layer in self.layers:
            if layer == self.base_layer:
                continue
            out += [f"{channel}.{layer}" for channel in self.channels]
        return out

    @classmethod
    def get_supported_filetypes(cls):
        r"""Get the supported file types in a dictionary.

        Returns
        -------
        dict: Mapping from file extension to class that should be used to
            read it.

        """
        out = {}
        if OpenEXR is not None:
            out['exr'] = cls
        return out

    def add_channels(self, im, z, layer="", sample=-1, window=None):
        r"""Add channels to the file from image data.

        Parameters
        ----------
        im: np.ndarray
            Image data of shape MxNx4 including RGBA channels.
        z:  np.ndarray
            Deepth data of shape MxN for pixels in im.
        layer: string, optional
            Name of the layer that the channels should be added to. If
            empty, the base layer will be assumed.
        sample: int, optional
            Index of sample that data represents.
        window: tuple, optional
            Bounds for the data in im & z in the image plane if different
            from the overall image bounds

        """
        if layer == self.base_layer:
            layer = ""
        if layer and not layer.startswith("."):
            layer = "." + layer
        if self.scale:
            max_z = 1.0
            z_mask = (z != np.inf)
            if self.sigma_clip is not None:
                max_im = im._clipping_value(self.sigma_clip)
                if z_mask.any():
                    max_z = max(z[z_mask].max(), 1.0)
            else:
                max_im = im[:, :, :3].max()
                if z_mask.any():
                    max_z = max(z[z_mask].max(), 1.0)
            alpha = im[:, :, 3]
            im = im[:, :, :3]
            if max_im != 0:
                im = np.clip(im[:, :, :3] / max_im, 0.0, 1.0)
            if max_z != 0:
                z = np.clip(z / max_z, 0.0, 1.0)
            im = np.concatenate([im, alpha[..., None]], axis=-1)
        for i, name in zip(range(im.shape[-1]), self.channels):
            self.fd.write(f"{name}{layer}", im[:, :, i],
                          sample=sample, window=window)
        if len(self.channels) > im.shape[-1]:
            self.fd.write(f"{self.channels[im.shape[-1]]}{layer}", z,
                          sample=sample, window=window)
