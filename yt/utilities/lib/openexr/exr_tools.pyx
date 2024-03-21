# distutils: include_dirs = OPENEXR_INC_DIR
# distutils: library_dirs = OPENEXR_LIB_DIR
# distutils: libraries = OPENEXR_LIBS
# distutils: depends = yt/utilities/lib/openexr/exr_tools.h
# distutils: language = c++
# distutils: extra_compile_args = CPP11_FLAG

import numpy as np
cimport numpy as np
from cython.operator cimport dereference
from libcpp.string cimport string
from libcpp.map cimport map


cdef class OutputFile:
    r"""A wrapper for OpenEXR output files.

    Args:
        filename (str): Name of the file that should be written to.
        displayShape (tuple): Shape of the image in the x and y
            directions (in pixels).
        channels (dict): Mapping between channel name and data type for
            channels that will be written to the file.
        bufferShape (tuple, optional): If provided, the shape of the
            buffer in the x and y direction that should be used for
            writing data to the file.
        tileShape (tuple, optional): If provided, the shape of tiles
            within the file in the x and y directions (in pixels).
        maxNSamples (int, optional): The maximum number of samples for
            each pixel in the file. Values greater than 1 indicate that
            the file is deep.
        compression (str, optional): The type of compression that should
            be used when writing the file.
        levelMode (str, optional): The level mode that should be used for
            a tiled file.

    Attributes:
        channelMap (dict): Mapping between channel name and data type.

    """

    cdef void _init_file(self, string filename,
                         map[string, PixelType] channelMap,
                         V2i displayDim, V2i bufferDim, V2i tileDim,
                         long maxNSamples, Compression compression,
                         LevelMode levelMode):
        self.c_file = new CPPOutputFile(filename, channelMap,
                                        displayDim, bufferDim, tileDim,
                                        maxNSamples, compression,
                                        levelMode)

    def __cinit__(self):
        # Initialize everything to NULL/0/None to prevent seg fault
        self.c_file = NULL

    def __init__(self, filename, displayShape, channels,
                 bufferShape=None, tileShape=None, maxNSamples=1,
                 compression=NO_COMPRESSION, levelMode=ONE_LEVEL):
        cdef map[string, PixelType] c_channelMap
        cdef string c_k
        cdef string c_filename = (<unicode>filename).encode('utf8')
        cdef V2i displayDim, bufferDim, tileDim
        if bufferShape is None:
            bufferShape = (0, 0)
        if tileShape is None:
            tileShape = (1, 1)
        self.channelMap = channels
        displayDim.x = displayShape[0]
        displayDim.y = displayShape[1]
        bufferDim.x = bufferShape[0]
        bufferDim.y = bufferShape[1]
        tileDim.x = tileShape[0]
        tileDim.y = tileShape[1]
        for k, v in self.channelMap.items():
            c_k = (<unicode>k).encode('utf8')
            if np.issubdtype(v, np.integer):
                self.channelMap[k] = np.uint32
                c_channelMap[c_k] = UINT
            else:
                self.channelMap[k] = np.float32
                c_channelMap[c_k] = FLOAT
        self._init_file(c_filename, c_channelMap,
                        displayDim, bufferDim, tileDim,
                        maxNSamples, compression, levelMode)

    def __dealloc__(self):
        if self.c_file != NULL:
            del self.c_file

    @property
    def dataWidth(self):
        r"""int: Width of the image data (in pixels)"""
        return self.c_file.dataWidth()

    @property
    def dataHeight(self):
        r"""int: Height of the image data (in pixels)"""
        return self.c_file.dataHeight()

    @property
    def dataSize(self):
        r"""int: Number of pixels in the image data"""
        return self.c_file.dataSize()

    @property
    def bufferWidth(self):
        r"""int: Width of the image buffer (in pixels)"""
        return self.c_file.bufferWidth()

    @property
    def bufferHeight(self):
        r"""int: Height of the image buffer (in pixels)"""
        return self.c_file.bufferHeight()

    @property
    def bufferSize(self):
        r"""int: Number of pixels in the image buffer"""
        return self.c_file.bufferSize()

    cdef void _write_UINT(self, string channelName,
                          np.ndarray[np.uint32_t, ndim=2] data,
                          Box2i window, long sample):
        self.c_file.write(channelName, &data[0,0], window, sample)

    cdef void _write_FLOAT(self, string channelName,
                           np.ndarray[np.float32_t, ndim=2] data,
                           Box2i window, long sample):
        self.c_file.write(channelName, &data[0,0], window, sample)

    def write(self, channelName, data, window=None, long sample=-1):
        r"""Set the data for a single channel in the file.

        Args:
            channelName (str): Name fo the channel to set the data for.
            data (np.ndarray): 2D contiguous array of data for the
                channel.
            window (tuple[tuple], optional): Tuple containing the minimum
                and maximum bounds of data in the x & y directions (in pixels).
            sample (int, optional): Index of the sample that the data
                corresponds to. Must be provided for deep files.

        """
        cdef string c_channelName = (<unicode>channelName).encode('utf8')
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be an ND array")
        cdef Box2i c_window
        if window is None:
            c_window = self.c_file.bufferWindow
        else:
            c_window.min.x = window[0][0]
            c_window.min.y = window[0][1]
            c_window.max.x = window[1][0]
            c_window.max.y = window[1][1]
        dtype = self.channelMap[channelName]
        if dtype == np.uint32:
            self._write_UINT(c_channelName, data.astype(dtype), c_window, sample)
        elif dtype == np.float32:
            self._write_FLOAT(c_channelName, data.astype(dtype), c_window, sample)

    # cdef unsigned int _writePixel_UINT(self, string channelName,
    #                                    unsigned int& data,
    #                                    long y, long x, long sample):
    #     self.c_file.writePixel[unsigned int](channelName, data, y, x, sample)

    # cdef float _writePixel_FLOAT(self, string channelName,
    #                              float data,
    #                              long y, long x, long sample):
    #     self.c_file.writePixel[float](channelName, data, y, x, sample)

    def writePixel(self, channelName, data, long y, long x, long sample=-1):
        r"""Write the data for a single pixel in a channel.

        Args:
            channelName (str): Name of channel to set data for.
            data (int, float): Value for the pixel.
            y (int): Index of the pixel to set in the y direction.
            x (int): Index of the pixel to set in the x direction.
            sample (int, optional): Index of the sample to set the data
                for in the pixel. Required for deep images and invalid
                for flat images.

        """
        cdef string c_channelName = (<unicode>channelName).encode('utf8')
        self.c_file.writePixel(channelName, data, y, x, sample)

    def setSampleCount(self, np.ndarray[np.uint32_t, ndim=2] N):
        r"""Set the number of samples at each pixel.

        Args:
            N (np.ndarray): Number of samples at each pixel.

        """
        cdef Box2i c_window = self.c_file.bufferWindow
        c_window.max.x = c_window.min.x + N.shape[0] - 1
        c_window.max.y = c_window.min.y + N.shape[0] - 1
        self.c_file.setSampleCount(&N[0,0], c_window, 0)

    def setSampleCountAll(self, unsigned int N):
        r"""Set the sample count for every pixel in the image to the
        same value.

        Args:
            N (int): Number of samples for every pixel.

        """
        self.c_file.setSampleCountAll(N, 0)



cdef class InputFile:
    r"""A wrapper for OpenEXR input files.

    Args:
        filename (str): Name of the file that should be read from.
        channels (list): List of channels that should be loaded from the
	    file.
        bufferShape (tuple, optional): If provided, the shape of the
            buffer in the x and y direction that should be used for
            reading data from the file.

    """

    cdef void _init_file(self, string filename,
                         vector[string] channelList, V2i bufferDim):
        self.c_file = new CPPInputFile(filename, channelList, bufferDim)
        cdef map[string, ChannelBuffer].iterator it = self.c_file.channels.begin()
        cdef string c_name
        cdef PixelType c_type
        while it != self.c_file.channels.end():
            c_name = dereference(it).first
            c_type = dereference(it).second.type
            name = (<bytes>c_name).decode('utf8')
            if c_type == UINT:
                self.channelMap[name] = np.uint32
            elif c_type == FLOAT:
                self.channelMap[name] = np.float32
            else:
                raise TypeError(f"Unsupported type: {c_type}")

    def __cinit__(self):
        # Initialize everything to NULL/0/None to prevent seg fault
        self.c_file = NULL

    def __init__(self, filename, channels=None, bufferShape=None):
        cdef vector[string] c_channelList
        cdef string c_k
        cdef string c_filename = (<unicode>filename).encode('utf8')
        cdef V2i bufferDim
        if bufferShape is None:
            bufferShape = (0, 0)
        self.channelMap = {}
        bufferDim.x = bufferShape[0]
        bufferDim.y = bufferShape[1]
        if channels is not None:
            for k in channels:
                c_k = (<unicode>k).encode('utf8')
                c_channelList.push_back(c_k)
        self._init_file(c_filename, c_channelList, bufferDim)

    def __dealloc__(self):
        if self.c_file != NULL:
            del self.c_file

    @property
    def dataWidth(self):
        r"""int: Width of the image data (in pixels)"""
        return self.c_file.dataWidth()

    @property
    def dataHeight(self):
        r"""int: Height of the image data (in pixels)"""
        return self.c_file.dataHeight()

    @property
    def dataSize(self):
        r"""int: Number of pixels in the image data"""
        return self.c_file.dataSize()

    @property
    def bufferWidth(self):
        r"""int: Width of the image buffer (in pixels)"""
        return self.c_file.bufferWidth()

    @property
    def bufferHeight(self):
        r"""int: Height of the image buffer (in pixels)"""
        return self.c_file.bufferHeight()

    @property
    def bufferSize(self):
        r"""int: Number of pixels in the image buffer"""
        return self.c_file.bufferSize()

    cdef np.ndarray[np.uint32_t, ndim=2] _read_UINT(self,
                                                    string channelName,
                                                    Box2i window,
                                                    long sample):
        cdef np.ndarray[np.uint32_t, ndim=2] data
        cdef V2i windowSize = window.size();
        data = np.empty((windowSize.x + 1, windowSize.y + 1), dtype=np.uint32)
        self.c_file.read(channelName, &data[0,0], window, sample)
        return data

    cdef np.ndarray[np.float32_t, ndim=2] _read_FLOAT(self,
                                                      string channelName,
                                                      Box2i window,
                                                      long sample):
        cdef np.ndarray[np.float32_t, ndim=2] data
        cdef V2i windowSize = window.size();
        data = np.empty((windowSize.x + 1, windowSize.y + 1), dtype=np.float32)
        self.c_file.read(channelName, &data[0,0], window, sample)
        return data

    def read(self, channelName, window=None, long sample=-1):
        r"""Get the data for a single channel in the file.

        Args:
            channelName (str): Name fo the channel to get the data for.
            window (tuple[tuple], optional): Tuple containing the minimum
                and maximum bounds of data that should be read in the
                x & y directions (in pixels).
            sample (int, optional): Index of the sample to get data for.
                Must be provided for deep files.

        Returns:
            np.ndarray: 2D contiguous array of data for the channel.

        """
        cdef string c_channelName = (<unicode>channelName).encode('utf8')
        cdef Box2i c_window
        if window is None:
            c_window = self.c_file.bufferWindow
        else:
            c_window.min.x = window[0][0]
            c_window.min.y = window[0][1]
            c_window.max.x = window[1][0]
            c_window.max.y = window[1][1]
        dtype = self.channelMap[channelName]
        if dtype == np.uint32:
            return self._read_UINT(c_channelName, c_window, sample)
        elif dtype == np.float32:
            return self._read_FLOAT(c_channelName, c_window, sample)

    cdef unsigned int _readPixel_UINT(self, string channelName,
                                      long y, long x, long sample):
        cdef unsigned int data
        data = self.c_file.readPixel[uint32_t](channelName, y, x, sample)
        return data

    cdef float _readPixel_FLOAT(self, string channelName,
                                long y, long x, long sample):
        cdef float data
        data = self.c_file.readPixel[float](channelName, y, x, sample)
        return data

    def readPixel(self, channelName, long y, long x, long sample=-1):
        r"""Read the data for a single pixel in a channel.

        Args:
            channelName (str): Name of channel to get data for.
            y (int): Index of the pixel to get in the y direction.
            x (int): Index of the pixel to get in the x direction.
            sample (int, optional): Index of the sample to get the data
                for in the pixel. Required for deep images and invalid
                for flat images.

        Returns:
            int, float: Value of pixel data.

        """
        cdef string c_channelName = (<unicode>channelName).encode('utf8')
        dtype = self.channelMap[channelName]
        if dtype == np.uint32:
            return self._readPixel_UINT(c_channelName, y, x, sample)
        elif dtype == np.float32:
            return self._readPixel_FLOAT(c_channelName, y, x, sample)

    cdef np.ndarray[np.uint32_t, ndim=2] _getSampleCount(self):
        cdef string sampleCountChannel = self.c_file.sampleCountChannelName()
        cdef Box2i window = self.c_file.bufferWindow
        return self._read_UINT(sampleCountChannel, window, -1)
        
    def getSampleCount(self):
        r"""Get the number of samples in each pixel of the image.

        Returns:
            np.ndarray: Number of samples at each pixel in the image.

        """
        return self._getSampleCount()

    def chunk(self, dim=None):
        r"""Yield an iterable chunk"""
        cdef V2i c_dim
        if dim is None:
            c_dim.x = 0
            c_dim.y = 0
        else:
            c_dim.x = dim[0]
            c_dim.y = dim[1]
        chunk = Chunk()
        chunk.c_chunk = self.c_file.chunk(c_dim)
        while not chunk.isComplete:
            yield chunk

cdef class Chunk:
    r"""A wrapper for an OpenEXR file chunk."""

    def __cinit__(self):
        self.c_chunk = CPPFileChunk()

    def __init__(self):
        pass

    @property
    def isDeep(self):
        r"""bool: True if chunk iteration is deep, false otherwise."""
        return self.c_chunk.isDeep()

    @property
    def isComplete(self):
        r"""bool: True if chunk iteration is complete, false otherwise."""
        return self.c_chunk.isComplete()

    def __getitem__(self, k):
        return self.read(k)

    cdef np.ndarray[np.uint32_t, ndim=2] _read_UINT(self,
                                                    string channelName,
                                                    long sample):
        cdef np.ndarray[np.uint32_t, ndim=2] data
        cdef V2i windowSize = self.c_chunk.window.size();
        data = np.empty((windowSize.x + 1, windowSize.y + 1), dtype=np.uint32)
        self.c_chunk.read(channelName, &data[0,0], sample)
        return data

    cdef np.ndarray[np.float32_t, ndim=2] _read_FLOAT(self,
                                                      string channelName,
                                                      long sample):
        cdef np.ndarray[np.float32_t, ndim=2] data
        cdef V2i windowSize = self.c_chunk.window.size();
        data = np.empty((windowSize.x + 1, windowSize.y + 1), dtype=np.float32)
        self.c_chunk.read(channelName, &data[0,0], sample)
        return data

    def read(self, channelName, long sample=-1):
        r"""Get the data for a single channel in the chunk.

        Args:
            channelName (str): Name fo the channel to get the data for.
            sample (int, optional): Index of the sample to get data for.
                Must be provided for deep files.

        Returns:
            np.ndarray: 2D contiguous array of data for the channel.

        """
        cdef string c_channelName = (<unicode>channelName).encode('utf8')
        cdef Box2i c_window
        dtype = self.channelMap[channelName]
        if sample == -1 and self.isDeep:
            return [self.read(channelName, sample=i) for i in
                    self.c_chunk.maxSampleCount()]
        if dtype == np.uint32:
            return self._read_UINT(c_channelName, sample)
        elif dtype == np.float32:
            return self._read_FLOAT(c_channelName, sample)

    cdef void _write_UINT(self, string channelName,
                          np.ndarray[np.uint32_t, ndim=2] data,
                          long sample):
        self.c_chunk.write(channelName, &data[0,0], sample)

    cdef void _write_FLOAT(self, string channelName,
                           np.ndarray[np.float32_t, ndim=2] data,
                           long sample):
        self.c_chunk.write(channelName, &data[0,0], sample)

    def write(self, channelName, data, long sample = -1):
        r"""Set the data for a single channel in the chunk.

        Args:
            channelName (str): Name fo the channel to set the data for.
            data (np.ndarray): 2D contiguous array of data for the
                channel.
            sample (int, optional): Index of the sample that the data
                corresponds to. Must be provided for deep files.

        """
        cdef string c_channelName = (<unicode>channelName).encode('utf8')
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be an ND array")
        dtype = self.channelMap[channelName]
        cdef V2i windowSize = self.c_chunk.window.size();
        assert windowSize.x == data.shape[0]
        assert windowSize.y == data.shape[1]
        if dtype == np.uint32:
            self._write_UINT(c_channelName, data.astype(dtype), sample)
        elif dtype == np.float32:
            self._write_FLOAT(c_channelName, data.astype(dtype), sample)
