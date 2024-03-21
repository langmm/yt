# cython: language_level=3
# distutils: language=c++

cimport numpy as np
from libc.stdint cimport uint8_t, uint32_t
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "Imath/ImathVec.h" namespace "Imath":
    cdef cppclass Vec2[T]:
        T x
        T y
        Vec2()
        Vec2(T a)
        Vec2(T a, T b)
    ctypedef Vec2[int] V2i

cdef extern from "Imath/ImathBox.h" namespace "Imath":
    cdef cppclass Box[V]:
        V min
        V max
        Box()
        Box(V)
        Box(V, V)
        V2i size()
    ctypedef Box[V2i] Box2i

cdef extern from "OpenEXR/ImfPixelType.h" namespace "Imf":
    cdef enum PixelType:
        UINT  = 0,
        HALF  = 1,
        FLOAT = 2,
        NUM_PIXELTYPES

cdef extern from "OpenEXR/ImfCompression.h" namespace "Imf":
    cdef enum Compression:
        NO_COMPRESSION = 0,     # no compression.
        RLE_COMPRESSION = 1,    # run length encoding.
        ZIPS_COMPRESSION = 2,   # zlib compression, one scan line at a time.
        ZIP_COMPRESSION = 3,    # zlib compression, in blocks of 16 scan lines.
        PIZ_COMPRESSION = 4,    # piz-based wavelet compression.
        PXR24_COMPRESSION = 5,  # lossy 24-bit float compression
        B44_COMPRESSION = 6,    # lossy 4-by-4 pixel block compression,
                                # fixed compression rate.
        B44A_COMPRESSION = 7,   # lossy 4-by-4 pixel block compression,
                                # flat fields are compressed more.
        DWAA_COMPRESSION = 8,   # lossy DCT based compression, in blocks
                                # of 32 scanlines. More efficient for partial
                                # buffer access.
        DWAB_COMPRESSION = 9,   # lossy DCT based compression, in blocks
                                # of 256 scanlines. More efficient space
                                # wise and faster to decode full frames
                                # than DWAA_COMPRESSION.
        NUM_COMPRESSION_METHODS # number of different compression methods.

cdef extern from "OpenEXR/ImfTileDescription.h" namespace "Imf":
    cdef enum LevelMode:
        ONE_LEVEL     = 0,
        MIPMAP_LEVELS = 1,
        RIPMAP_LEVELS = 2,
        NUM_LEVELMODES # number of different level modes

cdef extern from "exr_tools.h" namespace "OpenEXR":

    cdef cppclass FileBase:
        string filename
        void* file
        uint8_t flags
        Box2i displayWindow
        Box2i dataWindow
        Box2i bufferWindow
        map[string, ChannelBuffer] channels
        long nSampleAll

    cdef cppclass CPPFileChunk:
        void* file
        Box2i window
        long x
        long y
        long xmax
        long ymax
        CPPFileChunk()
        CPPFileChunk(FileBase*, V2i)
        CPPFileChunk(const CPPFileChunk&)
        bool isComplete()
        bool isDeep()
        CPPFileChunk& operator++()
        CPPFileChunk& operator--()
        unsigned int maxSampleCount()
        unsigned int minSampleCount()
        void read[T](string, T*, long)
        void write[T](string, T*, long)

    cdef cppclass ChannelBuffer:
        string name
        PixelType type
        uint8_t flags
        void* buffer
        Box2i dataWindow
        Box2i bufferWindow
        ChannelBuffer(string, PixelType, uint8_t, Box2i)

    cdef cppclass CPPOutputFile:
        string filename
        void* file
        uint8_t flags
        Box2i displayWindow
        Box2i dataWindow
        Box2i bufferWindow
        map[string, ChannelBuffer] channels
        long nSampleAll
        CPPOutputFile(string, map[string, PixelType], V2i, V2i, V2i, long, Compression, LevelMode) except +
        long dataWidth()
        long dataHeight()
        long dataSize()
        long bufferWidth()
        long bufferHeight()
        long bufferSize()
        CPPFileChunk chunk(V2i)
        string sampleCountChannelName()
        void setData[T](string, T, long, long, long)
        unsigned int maxSampleCount()
        unsigned int minSampleCount()
        void setSampleCount(unsigned int, long, long, bool)
        void setSampleCount(unsigned int*, Box2i, bool)
        void setSampleCountAll(unsigned int, bool)
        void write[T](string, T*, Box2i, long)
        void writeTile[T](string, T*, long, long, long)
        void writeLine[T](string, T*, long, long)
        void writePixel[T](string, T, long, long, long)

    cdef cppclass CPPInputFile:
        string filename
        void* file
        uint8_t flags
        Box2i displayWindow
        Box2i dataWindow
        Box2i bufferWindow
        map[string, ChannelBuffer] channels
        long nSampleAll
        CPPInputFile(string, vector[string], V2i)
        long dataWidth()
        long dataHeight()
        long dataSize()
        long bufferWidth()
        long bufferHeight()
        long bufferSize()
        CPPFileChunk chunk(V2i)
        string sampleCountChannelName()
        T* getDataPtr[T](string, long, long, long)
        unsigned int maxSampleCount()
        unsigned int minSampleCount()
        void read[T](string, T*, Box2i, long)
        void readTile[T](string, T*, long, long, long)
        void readLine[T](string, T*, long, long)
        void readPixel[T](string, T&, long, long, long)
        T readPixel[T](string, long, long, long)

cdef class OutputFile:
    cdef CPPOutputFile* c_file
    cdef object channelMap
    cdef void _init_file(self, string filename,
                         map[string, PixelType] channelMap,
                         V2i displayDim, V2i bufferDim, V2i tileDim,
                         long maxNSamples, Compression compression,
                         LevelMode levelMode)
    cdef void _write_UINT(self, string channelName,
                          np.ndarray[np.uint32_t, ndim=2] data,
                          Box2i window, long sample)
    cdef void _write_FLOAT(self, string channelName,
                           np.ndarray[np.float32_t, ndim=2] data,
                           Box2i window, long sample)

cdef class InputFile:
    cdef CPPInputFile* c_file
    cdef object channelMap
    cdef void _init_file(self, string filename,
                         vector[string] channelList, V2i bufferDim)
    cdef np.ndarray[np.uint32_t, ndim=2] _read_UINT(self,
                                                    string channelName,
                                                    Box2i window,
                                                    long sample)
    cdef np.ndarray[np.float32_t, ndim=2] _read_FLOAT(self,
                                                      string channelName,
                                                      Box2i window,
                                                      long sample)
    cdef unsigned int _readPixel_UINT(self, string channelName,
                                      long y, long x, long sample)
    cdef float _readPixel_FLOAT(self, string channelName,
                                long y, long x, long sample)
    cdef np.ndarray[np.uint32_t, ndim=2] _getSampleCount(self)

cdef class Chunk:
    cdef CPPFileChunk c_chunk
    cdef np.ndarray[np.uint32_t, ndim=2] _read_UINT(self,
                                                    string channelName,
                                                    long sample)
    cdef np.ndarray[np.float32_t, ndim=2] _read_FLOAT(self,
                                                      string channelName,
                                                      long sample)
    cdef void _write_UINT(self, string channelName,
                          np.ndarray[np.uint32_t, ndim=2] data,
                          long sample)
    cdef void _write_FLOAT(self, string channelName,
                           np.ndarray[np.float32_t, ndim=2] data,
                           long sample)
