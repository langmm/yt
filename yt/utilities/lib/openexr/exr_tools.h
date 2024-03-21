// TODO:
// - More complete handling of levels
// - Multilevel tiled
// - Multipart
// - Set partial buffer size in constructor and write/read incrementally
// - Allow addition of samples for partial pixels
#include <vector>
#include <string>
#include <utility>

#include <OpenEXR/ImfArray.h>

#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfPartType.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfTestFile.h>

#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfTiledInputFile.h>
#include <OpenEXR/ImfTiledOutputFile.h>

#include <OpenEXR/ImfDeepFrameBuffer.h>
#include <OpenEXR/ImfDeepScanLineInputFile.h>
#include <OpenEXR/ImfDeepScanLineOutputFile.h>
#include <OpenEXR/ImfDeepTiledInputFile.h>
#include <OpenEXR/ImfDeepTiledOutputFile.h>

using namespace std;

namespace OpenEXR {

  std::string _vector2str(const Imath::V2i& v) {
    return "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
  }
  std::string _window2str(const Imath::Box2i& window) {
    return "[" + _vector2str(window.min) + ", " + _vector2str(window.max) + "]";
  }
  Imath::V2i _windowSize(const Imath::Box2i& window) {
    Imath::V2i out = window.size();
    out.x++;
    out.y++;
    return out;
  }
  bool _window_contains(const Imath::Box2i& outside,
			const Imath::Box2i& inside) {
    return (inside.max.x >= inside.min.x &&
	    inside.max.y >= inside.min.y &&
	    outside.max.x >= outside.min.x &&
	    outside.max.y >= outside.min.y &&
	    inside.min.x >= outside.min.x &&
	    inside.max.x <= outside.max.x &&
	    inside.min.y >= outside.min.y &&
	    inside.max.y <= outside.max.y);
  }
  void _advance_window(const std::string& method,
		       const Imath::Box2i& bounds,
		       Imath::Box2i& window, bool dec=false,
		       bool ignore_error=false) {
    long xinc = 1;
    if (dec)
      xinc = -1;
    Imath::V2i value(window.max.x - window.min.x + 1,
		     window.max.y - window.min.y + 1);
    window.min.x += xinc * value.x;
    window.max.x += xinc * value.x;
    long yinc = 0;
    if (window.min.x < bounds.min.x) {
      yinc = -1;
      window.max.x = bounds.max.x;
      window.min.x = window.max.x - value.x + 1;
    } else if (window.max.x > bounds.max.x) {
      yinc = 1;
      window.min.x = bounds.min.x;
      window.max.x = window.min.x + value.x - 1;
    }
    window.min.y += yinc * value.y;
    window.max.y += yinc * value.y;
    if ((!ignore_error) &&
	(window.max.x > bounds.max.x ||
	 window.max.y > bounds.max.y ||
	 window.min.x < bounds.min.x ||
	 window.min.y < bounds.min.y))
      throw std::runtime_error(method + ": Updated window " + _window2str(window) + " is outside bounds " + _window2str(bounds));
  }

  enum EXRFLAG {
    EXRFLAG_INPUT        = 0x00001, //!< File is an input file
    EXRFLAG_OUTPUT       = 0x00002, //!< File is an output file
    EXRFLAG_TILED        = 0x00004, //!< File is tiled and tiles can be loaded in any order
    EXRFLAG_DEEP         = 0x00008, //!< File is deep with an arbitrary number of samples for each pixel
    EXRFLAG_MULTI        = 0x00010, //!< File is multipart
    EXRFLAG_SAMPLE_COUNT = 0x00020, //!< Channel stores the sample count
    EXRFLAG_PARTIAL      = 0x00040, //!< Channel buffer is only large enough to contain part of the total data in the channel
    EXRFLAG_BUFFER_DONE  = 0x00080, //!< Buffer has been loaded/written
  };

#define SWITCH_TYPE_NOERROR(METHOD, CASE_MACRO, TYPE_VAR)		\
  switch (TYPE_VAR) {							\
  case Imf::UINT: {							\
    CASE_MACRO(UINT, unsigned int);					\
    break;								\
  }									\
  case Imf::HALF: {							\
    CASE_MACRO(HALF, half);						\
    break;								\
  }									\
  case Imf::FLOAT: {							\
    CASE_MACRO(FLOAT, float);						\
    break;								\
  }									\
  default: {								\
    break;								\
  }									\
  }
#define SWITCH_TYPE(METHOD, CASE_MACRO, TYPE_VAR)			\
  switch (TYPE_VAR) {							\
  case Imf::UINT: {							\
    CASE_MACRO(UINT, unsigned int);					\
    break;								\
  }									\
  case Imf::HALF: {							\
    CASE_MACRO(HALF, half);						\
    break;								\
  }									\
  case Imf::FLOAT: {							\
    CASE_MACRO(FLOAT, float);						\
    break;								\
  }									\
  default: {								\
    throw std::runtime_error(#METHOD ": Unsupported type " + std::to_string(TYPE_VAR)); \
  }									\
  }

  //! Forward declaration
  class FileBase;
  class CPPFileChunk;

  /**
   * @brief Storage class for channels in the file
   */
  class ChannelBuffer {
  public:
    std::string name;    //!< Channel name
    Imf::PixelType type; //!< Enumerated type of data stored for pixels
    uint8_t flags;       //!< Bitwise EXRFLAG flags describing the channel
    void* buffer;        //!< Pointer to the beginning of the channel data stored in row major order
    Imath::Box2i dataWindow;     //!< Bounds of the data in the file
    Imath::Box2i bufferWindow;   //!< Bounds of the current buffer
    /**
     * @brief Empty constructor.
     */
    ChannelBuffer() :
      name(""), type(Imf::UINT), flags(0), buffer(0),
      dataWindow(), bufferWindow() {}
    /**
     * @brief Move constructor
     * @param[in,out] rhs Channel buffer to move
     */
    ChannelBuffer(ChannelBuffer&& rhs) :
      name(rhs.name), type(rhs.type),
      flags(rhs.flags), buffer(rhs.buffer),
      dataWindow(rhs.dataWindow), bufferWindow(rhs.bufferWindow) {
      rhs.buffer = nullptr;
    }
    /**
     * @brief Constructor
     * @param[in] nme Channel name
     * @param[in] typ Enumerated type of data stored for pixels
     * @param[in] flgs Bitwise EXRFLAG flags describing the channel
     * @param[in] dataWin Dimensions of the data for the channel
     *   in the file
     */
    ChannelBuffer(const std::string& nme, const Imf::PixelType& typ,
		  const uint8_t flgs,
		  const Imath::Box2i& dataWin) :
      name(nme), type(typ), flags(flgs), buffer(0),
      dataWindow(dataWin), bufferWindow() {}
    /**
     * @brief Destructor
     */
    ~ChannelBuffer() {
      if (buffer) {
#define DESTROY_BUFFER(EXR_TYPE, TYPE)					\
	if ((flags & EXRFLAG_DEEP) &&					\
	    !(flags & EXRFLAG_SAMPLE_COUNT)) {				\
	  Imf::Array2D<TYPE*>* tmp = (Imf::Array2D<TYPE*>*)buffer;	\
	  for (long y = 0; y < tmp->height(); y++) {			\
	    for (long x = 0; x < tmp->width(); x++) {			\
	      if ((*tmp)[y][x]) delete[] (*tmp)[y][x];			\
	      (*tmp)[y][x] = nullptr;					\
	    }								\
	  }								\
	  delete tmp;							\
	} else {							\
	  Imf::Array2D<TYPE>* tmp = (Imf::Array2D<TYPE>*)buffer;	\
	  delete tmp;							\
	}								\
	buffer = nullptr
	SWITCH_TYPE_NOERROR(~ChannelBuffer, DESTROY_BUFFER, type);
#undef DESTROY_BUFFER
      }
    }
  private:
    // INSPECTION UTILITIES
    /**
     * @brief Check if the buffer matches the expected size
     * @param[in] method Method name to add to error message.
     * @param[in] dataSize Size to compare against the buffer size
     * @param[in] expSize Expected size if different from the buffer size
     */
    void _checkSize(const std::string& method,
		    const long& dataSize,
		    long expSize=-1) const {
      if (expSize < 0)
	expSize = bufferSize();
      if (dataSize != expSize)
	throw std::runtime_error(method + ": dataSize (" +
				 std::to_string(dataSize) +
				 ") does not match the size expected by "
				 "the buffer (" +
				 std::to_string(expSize) + ")");
    }
    /**
     * @brief Check if a set of pixel coordinates are inside the current
     *   buffer window
     * @param[in] method Method name to add to error message.
     * @param[in] y Row of pixel
     * @param[in] x Column of pixel
     */
    void _checkIndex(const std::string& method,
		     const long y, const long x) const {
      Imath::V2i p(x, y);
      if (!bufferWindow.intersects(p)) {
	throw std::runtime_error(
	  method + ": Pixel at coordinates (" +
	  std::to_string(x) + ", " +
	  std::to_string(y) + ") is not inside "
	  "current buffer window [(" +
	  std::to_string(bufferWindow.min.x) + ", " +
	  std::to_string(bufferWindow.min.y) + "), (" +
	  std::to_string(bufferWindow.max.x) + ", " +
	  std::to_string(bufferWindow.max.y) + ")]");
      }
    }
    /**
     * @brief Check that a window is fully contained within the buffer
     * @param[in] method Method name to add to error message.
     * @param[in] window Window to check
     */
    void _checkWindow(const std::string& method,
		      const Imath::Box2i& window) const {
      if (!_window_contains(bufferWindow, window)) {
	throw std::runtime_error(method + ": Buffer window " +
				 _window2str(bufferWindow) +
				 " does not contain window " +
				 _window2str(window));
      }
    }
    /**
     * @brief Check that a sample index is valid
     * @param[in] method Method name to add to error message.
     * @param[in] s Sample index to check
     */
    void _checkSample(const std::string& method, const long s = -1) const {
      if (requiresSample()) {
	if (s < 0)
	  throw std::runtime_error(method + ": sample must be provided for deep files");
      } else {
	if (s > 0) {
	  if (flags & EXRFLAG_SAMPLE_COUNT)
	    throw std::runtime_error(method + ": sample invalid for the sample count buffer");
	  else
	    throw std::runtime_error(method + ": sample cannot be provided for flat files");
	}
      }
    }
    void _relativeIndex(const std::string& method,
			const long y, const long x,
			long& yrel, long& xrel) {
      _checkIndex(method, y, x);
      yrel = y - bufferWindow.min.y;
      xrel = x - bufferWindow.min.x;
    }
    /**
     * @brief Check if the assigned channel type matches the templated
     *   type.
     * @tparam T Type of data to check for match against the channel's
     *   enumerated type.
     * @param[in] method Method name to add to error message.
     */
    template<typename T>
    void _checkType(const std::string& method) const {
      throw std::runtime_error("checkType[" + method + "]: Unsupported type");
    }
    /**
     * @brief Check if the assigned channel type matches the templated
     *   type.
     * @tparam T Type of data to check for match against the channel's
     *   enumerated type.
     * @param[in] method Method name to add to error message.
     */
    template<typename T>
    void _checkBufferType(const std::string& method) const {
      throw std::runtime_error("checkType[" + method + "]: Unsupported type");
    }
#define CHECK_TYPE(EXR_TYPE, TYPE)					\
    template<>								\
    void _checkType<TYPE>(const std::string& method) const {		\
    if (type != Imf::EXR_TYPE)						\
      throw std::runtime_error("checkType[" + method + "]: Provided type (" #EXR_TYPE ") does not match the enumerated type (" + std::to_string(type) + ")"); \
    }									\
    template<>								\
    void _checkBufferType<TYPE*>(const std::string& method) const {	\
      if ((flags & EXRFLAG_SAMPLE_COUNT) || !(flags & EXRFLAG_DEEP)) {	\
	throw std::runtime_error("checkBufferType[" + method + "]: Pointers invalid for sampleCount channel and non-deep files"); \
      }									\
      if (type != Imf::EXR_TYPE)					\
	throw std::runtime_error("checkBufferType[" + method + "]: Provided type (" #EXR_TYPE ") does not match the enumerated type (" + std::to_string(type) + ")"); \
    }									\
    template<>								\
    void _checkBufferType<TYPE>(const std::string& method) const {	\
      if ((flags & EXRFLAG_DEEP) && !(flags & EXRFLAG_SAMPLE_COUNT)) {	\
	throw std::runtime_error("checkBufferType[" + method + "]: Pointers required for deep file channels"); \
      }									\
      if (type != Imf::EXR_TYPE)					\
	throw std::runtime_error("checkBufferType[" + method + "]: Provided type (" #EXR_TYPE ") does not match the enumerated type (" + std::to_string(type) + ")"); \
    }
    CHECK_TYPE(UINT, unsigned int)
    CHECK_TYPE(HALF, half)
    CHECK_TYPE(FLOAT, float)
#undef CHECK_TYPE
    // BUFFER UTILITIES
    /**
     * @brief Initialize the buffer to a 2D array with the channel's
     *   required size based on the template type T.
     * @tparam T Type of data in the initialized buffer.
     */
    template<typename T>
    void _initBuffer() {
      _checkBufferType<T>("initBuffer");
      Imf::Array2D<T>* data = nullptr;
      if (buffer) {
	data = (Imf::Array2D<T>*)buffer;
      } else {
	data = new Imf::Array2D<T>;
	buffer = (void*)data;
      }
      data->resizeErase(bufferHeight(), bufferWidth());
      buffer = (void*)data;
    }
    /**
     * @brief Get the pointer to the start of a section of data within
     *   the channel's buffer
     * @tparam T Type of data stored in the buffer
     * @returns Pointer to bufferWindow within the initialized channel
     *   buffer
     */
    template<typename T>
    char* _dataBuffer() {
      Imf::Array2D<T>& tmp = *((Imf::Array2D<T>*)buffer);
      return (char*)(&tmp[0][0]
		     - bufferWindow.min.x
		     - bufferWindow.min.y * bufferWidth());
    }
  public:
    /**
     * @brief Get the pointer to the start of a section of data within
     *   the channel's buffer
     * @returns Pointer to bufferWindow within the initialized channel
     *   buffer
     */
    char* dataBuffer() {
      char* out = 0;
#define DATA_BUFFER(EXR_TYPE, TYPE)					\
	if ((flags & EXRFLAG_DEEP) && !(flags & EXRFLAG_SAMPLE_COUNT))	\
	  out = _dataBuffer<TYPE*>();					\
	else								\
	  out = _dataBuffer<TYPE>()
      SWITCH_TYPE(dataBuffer, DATA_BUFFER, type);
#undef DATA_BUFFER
      return out;
    }
    void _updateFrameBuffer(Imath::Box2i newBufferWindow) {
      if (bufferWindow == newBufferWindow)
	return;
      bufferWindow = newBufferWindow;
      if (bufferWindow == dataWindow)
	flags &= ~EXRFLAG_PARTIAL;
      else
	flags |= EXRFLAG_PARTIAL;
#define INIT_BUFFER(EXR_TYPE, TYPE)					\
      if ((flags & EXRFLAG_DEEP) && !(flags & EXRFLAG_SAMPLE_COUNT)) {	\
	Imf::Array2D<TYPE*>* tmp = (Imf::Array2D<TYPE*>*)buffer;	\
	if (tmp) {							\
	  for (long y = 0; y < tmp->height(); y++) {			\
	    for (long x = 0; x < tmp->width(); x++) {			\
	      if ((*tmp)[y][x]) {					\
		delete[] (*tmp)[y][x];					\
		(*tmp)[y][x] = nullptr;					\
	      }								\
	    }								\
	  }								\
	}								\
	_initBuffer<TYPE*>();						\
	tmp = (Imf::Array2D<TYPE*>*)buffer;				\
	for (long y = 0; y < tmp->height(); y++) {			\
	  for (long x = 0; x < tmp->width(); x++) {			\
	    (*tmp)[y][x] = nullptr;					\
	  }								\
	}								\
      } else {								\
	_initBuffer<TYPE>();						\
      }
      SWITCH_TYPE(updateFrameBuffer, INIT_BUFFER, type);
#undef INIT_BUFFER
    }
    /**
     * @brief Update the buffer for the channel and add it to a frame buffer
     * @tparam fbClass Frame buffer type
     * @param[in] frameBuffer Frame buffer to add this channel to
     * @param[in] newBufferWindow Boundaries for the frame buffer that
     *   should be created.
     */
    template<typename fbClass>
    void updateFrameBuffer(fbClass&, Imath::Box2i) {
      throw std::runtime_error("updateFrameBuffer: Invalid frame buffer type");
    }
    template<>
    void updateFrameBuffer<Imf::FrameBuffer>(Imf::FrameBuffer& frameBuffer,
					     Imath::Box2i newBufferWindow) {
      _updateFrameBuffer(newBufferWindow);
      if (flags & EXRFLAG_DEEP)
	throw std::runtime_error("updateFrameBuffer: Deep files require Imf::DeepFrameBuffer buffers");
      frameBuffer.insert(
        name,
	Imf::Slice(
	  type,
	  dataBuffer(),
	  elementSize() * 1,               /* xStride for pointer array */
	  elementSize() * bufferWidth())); /* yStride for pointer array */
    }
    template<>
    void updateFrameBuffer<Imf::DeepFrameBuffer>(Imf::DeepFrameBuffer& frameBuffer,
						 Imath::Box2i newBufferWindow) {
      _updateFrameBuffer(newBufferWindow);
      if (!(flags & EXRFLAG_DEEP))
	throw std::runtime_error("updateFrameBuffer: Non-deep files require Imf::FrameBuffer buffers");
      if (flags & EXRFLAG_SAMPLE_COUNT) {
	frameBuffer.insertSampleCountSlice(
	  Imf::Slice(
	    type,
	    dataBuffer(),
	    elementSize() * 1,               /* xStride for pointer array */
	    elementSize() * bufferWidth())); /* yStride for pointer array */
      } else {
	frameBuffer.insert(
          name,
	  Imf::DeepSlice(
	    type,
	    dataBuffer(),
	    elementSize(1) * 1,              /* xStride for pointer array */
	    elementSize(1) * bufferWidth(),  /* yStride for pointer array */
	    elementSize() * 1));             /* stride for samples */
      }
    }

    /**
     * @brief Get the size of elements in the buffer.
     * @param[in] ptr If true, the size will be for pointers.
     * @returns Size of elements in bytes.
     */
    long elementSize(bool ptr = false) const {
#define ELEMENT_SIZE(EXR_TYPE, TYPE)			\
      if (ptr) {					\
	return sizeof(TYPE*);				\
      } else {						\
	return sizeof(TYPE);				\
      }
      SWITCH_TYPE(elementSize, ELEMENT_SIZE, type);
#undef ELEMENT_SIZE
    }

    /**
     * @brief Get the width of the data in the x direction (in pixels)
     * @returns Width
     */
    long dataWidth() const {
      return dataWindow.max.x - dataWindow.min.x + 1;
    }
    /**
     * @brief Get the height of the data in the y direction (in pixels)
     * @returns Height
     */
    long dataHeight() const {
      return dataWindow.max.y - dataWindow.min.y + 1;
    }
    /**
     * @brief Get the size of the data in pixels.
     * @returns Data size
     */
    long dataSize() const {
      return dataHeight() * dataWidth();
    }
    
    /**
     * @brief Get the width of the buffer in the x direction (in pixels)
     * @returns Width
     */
    long bufferWidth() const {
      return bufferWindow.max.x - bufferWindow.min.x + 1;
    }
    /**
     * @brief Get the height of the buffer in the y direction (in pixels)
     * @returns Height
     */
    long bufferHeight() const {
      return bufferWindow.max.y - bufferWindow.min.y + 1;
    }
    /**
     * @brief Get the size of the buffer in pixels.
     * @returns Buffer size
     */
    long bufferSize() const {
      return bufferHeight() * bufferWidth();
    }
    
    /**
     * @brief Determine if the data in the frame buffer is contiguous.
     * @returns true if the data is contiguous, false otherwise.
     */
    bool isContiguous() const {
      return ((flags & EXRFLAG_SAMPLE_COUNT) || !(flags & EXRFLAG_DEEP));
    }
    /**
     * @brief Determine if the channel requires a sample index.
     * @returns true if the data requies sample index, false otherwise.
     */
    bool requiresSample() const {
      return ((flags & EXRFLAG_DEEP) && !(flags & EXRFLAG_SAMPLE_COUNT));
    }
    
    // GET DATA POINTER
    /**
     * @brief Get the pointer to the memory containing the data for the
     *   pixel in row y and column x (and sample s for deep files).
     * @tparam T Type of data expected
     * @param[in] y Row of pixel to return the data pointer for.
     * @param[in] x Column of pixel to return the data pointer for.
     * @param[in] s Sample of pixel to return the data pointer for. Must
     *   be provided for deep files and cannot be provided for scanline
     *   files.
     * @returns Data pointer
     */
    template<typename T>
    T* getDataPtr(const long, const long, const long = -1) {
      throw std::runtime_error("getDataPtr: Unsupported data type");
      return nullptr;
    }
    /**
     * @brief Get a const pointer to the memory containing the data for
     *   the pixel in row y and column x (and sample s for deep files).
     * @tparam T Type of data expected
     * @param[in] y Row of pixel to return the data pointer for.
     * @param[in] x Column of pixel to return the data pointer for.
     * @param[in] s Sample of pixel to return the data pointer for. Must
     *   be provided for deep files and cannot be provided for scanline
     *   files.
     * @returns Constant data pointer
     */
    template<typename T>
    const T* getDataPtr(const long y, const long x,
			const long s = -1) const {
      return const_cast<ChannelBuffer*>(this)->getDataPtr<T>(y, x, s);
    }
#define GET_DATA(EXR_TYPE, TYPE)					\
    template<>								\
    TYPE* getDataPtr<TYPE>(const long y, const long x,			\
			   const long s) {				\
      _checkType<TYPE>("getDataPtr");					\
      _checkSample("getDataPtr", s);					\
      if (!buffer)							\
	throw std::runtime_error("getDataPtr: Buffer not initialized");	\
      TYPE* data = nullptr;						\
      long yrel = 0, xrel = 0;						\
      _relativeIndex("getDataPtr_" #EXR_TYPE, y, x, yrel, xrel);	\
      if ((flags & EXRFLAG_DEEP) && !(flags & EXRFLAG_SAMPLE_COUNT)) {	\
	Imf::Array2D<TYPE*>* tmp = (Imf::Array2D<TYPE*>*)buffer;	\
	if (!((*tmp)[yrel][xrel]))					\
	  throw std::runtime_error("getDataPtr: Sample buffer not initialized"); \
	data = &((*tmp)[yrel][xrel][s]);				\
      } else {								\
	Imf::Array2D<TYPE>* tmp = (Imf::Array2D<TYPE>*)buffer;		\
	data = &((*tmp)[yrel][xrel]);					\
      }									\
      return data;							\
    }
    GET_DATA(UINT, unsigned int)
    GET_DATA(HALF, half)
    GET_DATA(FLOAT, float)
#undef GET_DATA
  
    // GET DATA
    /**
     * @brief Get the data for the pixel in row y and column x (and
     *   sample s for deep files).
     * @tparam T Type of data expected
     * @param[out] data Reference to set to the pixel data.
     * @param[in] y Row of pixel to return the data for.
     * @param[in] x Column of pixel to return the data for.
     * @param[in] s Sample of pixel to return the data for. Must
     *   be provided for deep files and cannot be provided for scanline
     *   files.
     */
    template<typename T>
    void getData(T& data, const long y, const long x,
		 const long s = -1) const {
      const T* tmp = getDataPtr<T>(y, x, s);
      if (!tmp)
	throw std::runtime_error("getData: Data pointer is null");
      data = tmp[0];
    }
    /**
     * @brief Get the data for the pixel in row y and column x (and
     *   sample s for deep files).
     * @tparam T Type of data expected
     * @param[in] y Row of pixel to return the data for.
     * @param[in] x Column of pixel to return the data for.
     * @param[in] s Sample of pixel to return the data for. Must
     *   be provided for deep files and cannot be provided for scanline
     *   files.
     * @returns Pixel data.
     */
    template<typename T>
    T getData(const long y, const long x, const long s = -1) const {
      T out = 0;
      getData<T>(out, y, x, s);
      return out;
    }
       
    // SET DATA
    /**
     * @brief Set the data for a given window in the channel from a given
     *   buffer.
     * @tparam T Type of data.
     * @param[in] data Buffer containing channel data for the window.
     * @param[in] window Bounds of pixels that data should be assigned to.
     * @param[in] s Sample to assign data for. Must be provided for deep
     *   files and cannot be provided for scanline files.
     */
    template<typename T>
    void setData(const T* data, const Imath::Box2i window,
		 const long s = -1) {
      _checkWindow("setData", window);
      _checkSample("setData", s);
      Imath::V2i windowSize = _windowSize(window);
      if (isContiguous() && window == bufferWindow) {
	T* dst = getDataPtr<T>(0, 0, s);
	memcpy(dst, data, windowSize.x * windowSize.y * sizeof(T));
      } else {
	for (long y = 0; y < windowSize.y; y++) {
	  for (long x = 0; x < windowSize.x; x++) {
	    setData(data[y * windowSize.x + x],
		    y + window.min.y, x + window.min.x, s);
	  }
	}
      }
    }
    /**
     * @brief Set the data for the pixel in row y and column x (and
     *   sample s for deep files).
     * @tparam T Type of data.
     * @param[in] data Pixel data.
     * @param[in] y Row of pixel to assign.
     * @param[in] x Column of pixel to assign.
     * @param[in] s Sample of pixel to assign. Must be provided for deep
     *   files and cannot be provided for scanline files.
     */
    template<typename T>
    void setData(const T& data, const long y, const long x,
		 const long s = -1) {
      T* dst = getDataPtr<T>(y, x, s);
      if (!dst)
	throw std::runtime_error("setData: Data pointer is null");
      dst[0] = data;
    }

    // COPY
    /**
     * @brief Copy data inside a given window from the channel into the
     *   provided buffer.
     * @tparam T Type of data.
     * @param[in,out] data Pre-allocated buffer that channel data should
     *     be copied into.
     * @param[in] window Bounds of data that should be returned. If not
     *   provided, the entire buffer will be assumed.
     * @param[in] s Sample to copy data for in a deep file. Must be
     *   provided for deep files and cannot be provided for scanline
     *   files.
     */
    template<typename T>
    void copyData(T* data, const Imath::Box2i window,
		  const long s = -1) const {
      _checkWindow("copyData", window);
      _checkSample("copyData", s);
      Imath::V2i windowSize = _windowSize(window);
      if (isContiguous() && window == bufferWindow) {
	const T* src = getDataPtr<T>(0, 0, s);
	memcpy(data, src, windowSize.x * windowSize.y * sizeof(T));
      } else {
	for (long y = 0; y < windowSize.y; y++) {
	  for (long x = 0; x < windowSize.x; x++) {
	    data[y * windowSize.x + x] = getData<T>(y + window.min.y,
						    x + window.min.x,
						    s);
	  }
	}
      }
    }

    // SAMPLE COUNT
    /**
     * @brief Set the sample count for all pixels, reallocating the
     *   buffer as necessary.
     * @param[in] N Number of samples for each pixel in the channel in
     *   row-major order.
     * @param[in] window Bounds of pixels that data should be assigned to.
     */
    void setSampleCount(const unsigned int* N,
			const Imath::Box2i window) {
      if (!buffer)
	throw std::runtime_error("setSampleCount: Buffer is not initialized");
      if (!(flags & EXRFLAG_DEEP))
	throw std::runtime_error("setSampleCount: Cannot set sample count for file that is not deep");
      _checkWindow("setSampleCount", window);
      if (flags & EXRFLAG_SAMPLE_COUNT) {
	setData(N, window);
	return;
      }
      Imath::V2i windowSize = _windowSize(window);
      for (long y = 0; y < windowSize.y; y++) {
	for (long x = 0; x < windowSize.x; x++) {
	  setSampleCount(N[y * windowSize.x + x],
			 window.min.y + y,
			 window.min.x + x);
	}
      }
    }
    /**
     * @brief Set the sample count for a single pixel, reallocating the
     *   channel as necessary to contain that many samples.
     * @param[in] N Number of samples for the pixel.
     * @param[in] y Row of pixel to set the sample count for.
     * @param[in] x Column of pixel to set the sample count for.
     */
    void setSampleCount(const unsigned int& N,
			const long y, const long x) {
      if (!buffer)
	throw std::runtime_error("setSampleCount: Buffer is not initialized");
      if (!(flags & EXRFLAG_DEEP))
	throw std::runtime_error("setSampleCount: Cannot set sample count for file that is not deep");
      if (flags & EXRFLAG_SAMPLE_COUNT) {
	setData(N, y, x);
	return;
      }
      // TODO: Allow existing sample count to be modified?
#define SET_SAMPLE(EXR_TYPE, TYPE)					\
      Imf::Array2D<TYPE*>* tmp = (Imf::Array2D<TYPE*>*)buffer;		\
      if ((*tmp)[y][x])							\
	throw std::runtime_error("setSampleCount: Sample pointer is already initialized"); \
      (*tmp)[y][x] = new TYPE[N]
      SWITCH_TYPE(setSampleCount, SET_SAMPLE, type);
#undef SET_SAMPLE
    }

#define FILE_METHOD_BASE(type, method, args_def, args, deepcls, normcls, error_chk, valid_typ, ret, body) \
    type method args_def {						\
    if (error_chk) {							\
	throw std::runtime_error(#method ": Only valid for " #valid_typ " files"); \
      }									\
      body;								\
      if (flags & EXRFLAG_DEEP) {					\
	ret ((Imf::deepcls*)file)->method args;				\
      } else {								\
	ret ((Imf::normcls*)file)->method args;				\
      }									\
    }
#define FILE_METHOD(type, method, args_def, args, deepcls, normcls, error_chk, valid_typ, body) \
    FILE_METHOD_BASE(type, method, args_def, args, deepcls, normcls, error_chk, valid_typ, return, body)
#define FILE_METHOD_NORET(method, args_def, args, deepcls, normcls, error_chk, valid_typ, body) \
    FILE_METHOD_BASE(void, method, args_def, args, deepcls, normcls, error_chk, valid_typ, , body)
  };    
    
  /**
   * @brief Base class for reading & writing OpenEXR files
   */
  class FileBase {
  public:
    std::string filename; //!< Full path to the file
    void* file;           //!< Pointer to the underlying OpenEXR file
    uint8_t flags;        //!< Bitwise EXRFLAG flags describing the file
    Imath::Box2i displayWindow; //!< Dimensions of the display in the file
    Imath::Box2i dataWindow;    //!< Dimensions of the data in the file
    Imath::Box2i bufferWindow;  //!< Dimensions of the current frame buffer
    std::map<std::string, ChannelBuffer> channels; //!< Mapping of channels in the file
    long nSampleAll;      //!< Global number of samples
    /**
     * @brief Constructor
     * @param[in] fname Full path to the file
     */
    FileBase(const std::string& fname) :
      filename(fname), file(nullptr), flags(0),
      displayWindow(), dataWindow(), bufferWindow(),
      channels(), nSampleAll(-1) {}
    /**
     * @brief Destructor
     */
    virtual ~FileBase() {}
    /**
     * @brief Destroy the underlying OpenEXR file
     * @tparam fileClass File class to destory
     */
    template<typename fileClass>
    void destroyFile() {
      if (file) {
	fileClass* tmp = (fileClass*)file;
	delete tmp;
	file = nullptr;
      }
    }
    /**
     * @brief Check if the chunk is deep
     * @returns true if the chunk is deep, false otherwise
     */
    bool isDeep() const {
      return (flags & EXRFLAG_DEEP);
    }

  private:

    /**
     * @brief Check that a sample index is valid
     * @param[in] method Method name to add to error message.
     * @param[in] s Sample index to check
     */
    void _checkSample(const std::string& method,
		      const long s = -1) const {
      if (flags & EXRFLAG_DEEP) {
	if (s < 0)
	  throw std::runtime_error(method + ": sample must be provided for deep files");
      } else {
	if (s > 0)
	  throw std::runtime_error(method + ": sample cannot be provided for flat files");
      }
    }
    /**
     * @brief Check that a sample index is valid
     * @param[in] method Method name to add to error message.
     * @param[in] y index of pixel in the y durection to check sample
     *   index against
     * @param[in] x index of pixel in the x durection to check sample
     *   index against
     * @param[in] s Sample index to check
     */
    void _checkSample(const std::string& method,
		      const long y, const long x,
		      const long s = -1) const {
      _checkSample(method, s);
      if ((flags & EXRFLAG_DEEP) && (s >= getSampleCount(y, x)))
	throw std::runtime_error(method + ": sample index (" + std::to_string(s) + ") exceeds sample count (" + std::to_string(getSampleCount(y, x)) + ") for pixel [" + std::to_string(x) + ", " + std::to_string(y) + "]");
    }
    /**
     * @brief Check that a sample index is valid
     * @param[in] method Method name to add to error message.
     * @param[in] window Bounds to check sample index within. If not
     *   provided, the entire buffer will be assumed.
     * @param[in] s Sample index to check
     */
    void _checkSample(const std::string& method,
		      Imath::Box2i window=Imath::Box2i(),
		      const long s = -1) const {
      _checkSample(method, s);
      if (window.isEmpty())
	window = bufferWindow;
      if ((flags & EXRFLAG_DEEP) && (s >= maxSampleCount(window)))
	throw std::runtime_error(method + ": sample index (" + std::to_string(s) + ") max exceeds sample count (" + std::to_string(maxSampleCount(window)) + ") for window " + _window2str(window));
    }
    
  public:
    
    /**
     * @brief Get the width of the data in the x direction (in pixels)
     * @returns Width
     */
    long dataWidth() const {
      return dataWindow.max.x - dataWindow.min.x + 1;
    }
    /**
     * @brief Get the height of the data in the y direction (in pixels)
     * @returns Height
     */
    long dataHeight() const {
      return dataWindow.max.y - dataWindow.min.y + 1;
    }
    /**
     * @brief Get the size of the data in pixels.
     * @returns Data size
     */
    long dataSize() const {
      return dataHeight() * dataWidth();
    }
    
    /**
     * @brief Get the width of the buffer in the x direction (in pixels)
     * @returns Width
     */
    long bufferWidth() const {
      return bufferWindow.max.x - bufferWindow.min.x + 1;
    }
    /**
     * @brief Get the height of the buffer in the y direction (in pixels)
     * @returns Height
     */
    long bufferHeight() const {
      return bufferWindow.max.y - bufferWindow.min.y + 1;
    }
    /**
     * @brief Get the size of the buffer in pixels.
     * @returns Buffer size
     */
    long bufferSize() const {
      return bufferHeight() * bufferWidth();
    }

    /**
     * @brief Get the channel buffer containing the requested channel.
     * @param[in] channelName Name of the channel to get the buffer for
     * @returns Channel buffer
     */
    ChannelBuffer& getChannel(const std::string& channelName) {
      std::map<std::string, ChannelBuffer>::iterator it = channels.find(channelName);
      if (it == channels.end())
	throw std::runtime_error("getChannel: Could not locate channel \"" + channelName + "\"");
      return it->second;
    }
    /**
     * @brief Get a constant reference to the channel buffer containing
     *   the requested channel.
     * @param[in] channelName Name of the channel to get the buffer for
     * @returns Const channel buffer
     */
    const ChannelBuffer& getChannel(const std::string& channelName) const {
      return const_cast<FileBase*>(this)->getChannel(channelName);
    }
    
    /**
     * @brief Get the channel buffer containing the sample count (only
     *   valid for deep files).
     * @returns Sample count channel buffer.
     */
    ChannelBuffer& sampleCountChannel() {
      for (std::map<std::string, ChannelBuffer>::iterator it = channels.begin();
	   it != channels.end(); it++) {
	if (it->second.flags & EXRFLAG_SAMPLE_COUNT) {
	  return it->second;
	}
      }
      throw std::runtime_error("sampleCountChannel: Failed to locate a channel containing sample counts");
    }
    /**
     * @brief Get a constant reference to the channel buffer containing
     *   the sample count (only valid for deep files).
     * @returns Sample count channel buffer.
     */
    const ChannelBuffer& sampleCountChannel() const {
      return const_cast<FileBase*>(this)->sampleCountChannel();
    }
    /**
     * @brief Get the name of the channel buffer containing the sample
     *   count.
     * @returns Sample count channel name.
     */
    std::string sampleCountChannelName() const {
      return sampleCountChannel().name;
    }
    /**
     * @brief Get the header for the file
     * @returns Header reference
     */
    virtual const Imf::Header& header() const {
      throw std::runtime_error("header: Virtual method must be overriden");
    }
    /**
     * @brief Add a new channel to the file
     * @param[in] channelName Name of the channel to add
     * @param[in] channelType Type of data that the new channel will
     *   hold.
     * @param[in] isSampleCount If true, the channel is not a real
     *   a "real" channel and will be used to store the count of samples
     *   at each pixel in a deep file.
     */
    void addChannel(const std::string& channelName,
		    const Imf::PixelType& channelType,
		    const bool isSampleCount = false) {
      uint8_t channelFlags = flags;
      if (isSampleCount)
	channelFlags |= EXRFLAG_SAMPLE_COUNT;
      channels.emplace(std::piecewise_construct,
		       std::forward_as_tuple(channelName),
		       std::forward_as_tuple(channelName,
					     channelType,
					     channelFlags,
					     dataWindow));
    }
    /**
     * @brief Add the channels to the image.
     * @param[in] header Header describing the OpenEXR file
     * @param[in] channelList Names of channels to create buffers for.
     *   If empty, all of the channels will be included.
     */
    void addChannels(const Imf::Header& header,
		     const std::vector<std::string>& channelList = {}) {
      if (flags & EXRFLAG_DEEP) {
	addChannel("sampleCount", Imf::UINT, true);
      }
      if (channelList.empty()) {
	for (Imf::ChannelList::ConstIterator it = header.channels().begin();
	     it != header.channels().end(); it++) {
	  addChannel(it.name(), it.channel().type);
	}
      } else {
	for (std::vector<std::string>::const_iterator it = channelList.begin();
	     it != channelList.end(); it++) {
	  addChannel(*it, header.channels()[*it].type);
	}
      }
    }
    /**
     * @brief Create and assign a new frame buffer
     * @tparam fileClass Class of file that frame buffer is being added to
     * @tparam fbClass Class of frame buffer to create
     * @param[in] newBufferWindow Boundaries for the frame buffer that
     *   should be created.
     */
    template<typename fileClass, typename fbClass>
    void _updateFrameBuffer(Imath::Box2i newBufferWindow=Imath::Box2i()) {
      if (!file)
	throw std::runtime_error("updateFrameBuffer: File not initialized");
      if (newBufferWindow.isEmpty())
	newBufferWindow = dataWindow;
      if (newBufferWindow == bufferWindow)
	return;
      bufferWindow = newBufferWindow;
      fbClass frameBuffer;
      for (std::map<std::string, ChannelBuffer>::iterator it = channels.begin();
	   it != channels.end(); it++) {
	it->second.updateFrameBuffer(frameBuffer, bufferWindow);
      }
      ((fileClass*)file)->setFrameBuffer(frameBuffer);
      flags &= ~EXRFLAG_BUFFER_DONE;
      if (nSampleAll >= 0)
	setSampleCountAll(nSampleAll);
    }
    /**
     * @brief Create and assign a new frame buffer
     * @param[in] newBufferWindow Boundaries for the frame buffer that
     *   should be created.
     */
    virtual void updateFrameBuffer(Imath::Box2i=Imath::Box2i()) {
      throw std::runtime_error("updateFrameBuffer: Virtual method must be overridden");
    }
    /**
     * @brief Create and assign a new frame buffer
     * @param[in] bufferDim Dimensions of the new buffer (in pixels)
     */
    Imath::Box2i initFrameBuffer(Imath::V2i bufferDim) {
      if (bufferDim.x == 0 || bufferDim.y == 0) {
	if (flags & EXRFLAG_PARTIAL) {
	  if (flags & EXRFLAG_TILED) {
	    bufferDim.x = header().tileDescription().xSize;
	    bufferDim.y = header().tileDescription().ySize;
	  } else {
	    bufferDim.x = dataWidth();
	    bufferDim.y = 1;
	  }
	} else {
	  bufferDim.x = dataWidth();
	  bufferDim.y = dataHeight();
	}
      }
      if (bufferDim.x > dataWidth())
	throw std::runtime_error("initFrameBuffer: buffer dimensions exceed the data dimensions in the x direction");
      if (bufferDim.y > dataHeight())
	throw std::runtime_error("initFrameBuffer: buffer dimensions exceed the data dimensions in the y direction");
      Imath::Box2i newBufferWindow;
      newBufferWindow.min.x = dataWindow.min.x;
      newBufferWindow.min.y = dataWindow.min.y;
      newBufferWindow.max.x = dataWindow.min.x + bufferDim.x - 1;
      newBufferWindow.max.y = dataWindow.min.y + bufferDim.y - 1;
      return newBufferWindow;
    }
    /**
     * @brief Increment the frame buffer in row-major order.
     */
    void advanceFrameBuffer(bool dec=false) {
      Imath::Box2i newBufferWindow(bufferWindow);
      _advance_window("advanceFrameBuffer: ", dataWindow,
		      newBufferWindow, dec);
      this->updateFrameBuffer(newBufferWindow);
    }
  
    // GET DATA POINTER
    /**
     * @brief Get the the pointer to data in a single pixel in a channel
     * @tparam T Type of data expected
     * @param[in] channelName Name of channel to get pointer for
     * @param[in] x Index of pixel in the x direction to get pointer for
     * @param[in] y Index of pixel in the y direction to get pointer for
     * @param[in] s Sample to get pointer for in a deep file. Required
     *   for deep files, but not allowed for flat files.
     * @returns Data pointer
     */
    template<typename T>
    T* getDataPtr(const std::string& channelName,
		  const long y, const long x,
		  const long s = -1) {
      _checkSample("getDataPtr", y, x, s);
      return getChannel(channelName).getDataPtr<T>(y, x, s);
    }
  
    // SAMPLE COUNT
    /**
     * @brief Get the maximum sample count within a window of the buffer
     * @param[in] window Window to get maximum sample count inside. If not
     *   provided, the entire buffer window will be used.
     * @returns Maximum sample count
     */
    unsigned int maxSampleCount(Imath::Box2i window=Imath::Box2i()) const {
      if (!isDeep())
	throw std::runtime_error("FileBase::maxSampleCount: File is not deep");
      if (window.isEmpty())
	window = bufferWindow;
      unsigned int out = 0;
      for (long y = window.min.y; y < window.max.y + 1; y++) {
	for (long x = window.min.x; x < window.max.x + 1; x++) {
	  out = std::max(out, getSampleCount(y, x));
	}
      }
      return out;
    }
    /**
     * @brief Get the minimum sample count within a window of the buffer
     * @param[in] window Window to get minimum sample count inside. If not
     *   provided, the entire buffer window will be used.
     * @returns Minimum sample count
     */
    unsigned int minSampleCount(Imath::Box2i window=Imath::Box2i()) const {
      if (!isDeep())
	throw std::runtime_error("FileBase::minSampleCount: File is not deep");
      if (window.isEmpty())
	window = bufferWindow;
      unsigned int out = getSampleCount(0, 0);
      for (long y = window.min.y; y < window.max.y + 1; y++) {
	for (long x = window.min.x; x < window.max.x + 1; x++) {
	  out = std::min(out, getSampleCount(y, x));
	}
      }
      return out;
    }
    /**
     * @brief Get the buffer containing sample counts for the deep file.
     * @returns Pointer to the memory containing the sample counts for
     *   each pixel in row-major order.
     */
    unsigned int* getSampleCount() {
      unsigned int * N = sampleCountChannel().getDataPtr<unsigned int>(0, 0);
      if (!N)
	throw std::runtime_error("getSampleCount: Failed to locate a channel containing sample counts");
      return N;
    }
    /**
     * @brief Get the sample count for a single pixel
     * @param[in] y Row of pixel to get the sample count for.
     * @param[in] x Column of pixel to get the sample count for.
     * @returns Sample count for the requested pixel
     */
    unsigned int getSampleCount(const long y, const long x) const {
      return sampleCountChannel().getData<unsigned int>(y, x);
    }
    /**
     * @brief Set the sample count for a single pixel, reallocating the
     *   channels as necessary to contain that many samples.
     * @param[in] N Number of samples for the pixel.
     * @param[in] y Row of pixel to set the sample count for.
     * @param[in] x Column of pixel to set the sample count for.
     * @param[in] skipSampleCount If true, don't set the sample count on
     *   the channel used to stored the sample count buffer (e.g. if the
     *   sample count was read from the file).
     */
    void setSampleCount(const unsigned int& N,
			const long y, const long x,
			bool skipSampleCount=false) {
      if (!(flags & EXRFLAG_OUTPUT))
	std::runtime_error("FileBase::setSampleCount: Output file required");
      if (nSampleAll >= 0 && N != nSampleAll)
	throw std::runtime_error("FileBase::setSampleCount:"
				 " Existing global sample count (" +
				 std::to_string(nSampleAll) + ") does"
				 " not match new local sample count (" +
				 std::to_string(N) + ")");
      for (std::map<std::string, ChannelBuffer>::iterator it = channels.begin();
	   it != channels.end(); it++) {
	if (skipSampleCount && (it->second.flags & EXRFLAG_SAMPLE_COUNT))
	  continue;
	it->second.setSampleCount(N, y, x);
      }
    }
    /**
     * @brief Set the sample count for each pixel, reallocating the
     *   channels as necessary to contain that many samples.
     * @param[in] N Number of samples for each pixel in row-major order.
     * @param[in] window Bounds of pixels that data should be assigned to.
     * @param[in] skipSampleCount If true, don't set the sample count on
     *   the channel used to stored the sample count buffer (e.g. if the
     *   sample count was read from the file).
     */
    void setSampleCount(const unsigned int* N,
			const Imath::Box2i window,
			bool skipSampleCount=false) {
      if (!(flags & EXRFLAG_OUTPUT))
	std::runtime_error("FileBase::setSampleCount: Output file required");
      if (nSampleAll >= 0) {
	Imath::V2i windowSize = _windowSize(window);
	for (long y = 0; y < windowSize.y; y++) {
	  for (long x = 0; x < windowSize.x; x++) {
	    if (N[y * windowSize.x + x] != nSampleAll)
	      throw std::runtime_error("FileBase::setSampleCount:"
				       " Existing global sample count (" +
				       std::to_string(nSampleAll) + ") does"
				       " not match new local sample count (" +
				       std::to_string(N[y * windowSize.x + x]) +
				       ") for pixel " +
				       _vector2str(Imath::V2i(x, y)));
	    
	  }
	}
	for (std::map<std::string, ChannelBuffer>::iterator it = channels.begin();
	     it != channels.end(); it++) {
	  if (skipSampleCount && (it->second.flags & EXRFLAG_SAMPLE_COUNT))
	    continue;
	  it->second.setSampleCount(N, window);
	}
      }
    }
    /**
     * @brief Set the sample count for all pixels to be the same,
     *   reallocating the channels as necessary to contain that many
     *   samples.
     * @param[in] N Number of samples for the pixels.
     * @param[in] skipSampleCount If true, don't set the sample count on
     *   the channel used to stored the sample count buffer (e.g. if the
     *   sample count was read from the file).
     */
    void setSampleCountAll(const unsigned int& N,
			   bool skipSampleCount=false) {
      if (!(flags & EXRFLAG_OUTPUT))
	std::runtime_error("FileBase::setSampleCountAll: Output file required");
      if (nSampleAll < 0)
	nSampleAll = N;
      else if (N != nSampleAll)
	throw std::runtime_error("FileBase::setSampleCountAll:"
				 " Existing global sample count (" +
				 std::to_string(nSampleAll) + ") does"
				 " not match new global sample count (" +
				 std::to_string(N) + ")");
      for (long y = 0; y < bufferHeight(); y++) {
	for (long x = 0; x < bufferWidth(); x++) {
	  setSampleCount(N,
			 y + bufferWindow.min.y,
			 x + bufferWindow.min.x,
			 skipSampleCount);
	}
      }
    }

    // READ
    /**
     * @brief Read data from the file.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to copy data from.
     * @param[in] data Preallocated buffer that channel data should be
     *   copied into.
     * @param[in] window Bounds of data that should be returned. If not
     *   provided, the entire buffer will be assumed.
     * @param[in] s Sample to copy data for in a deep file. Must be
     *   provided for deep files and cannot be provided for scanline
     *   files.
     */
    template<typename T>
    void read(const std::string& channelName, T* data,
	      Imath::Box2i window=Imath::Box2i(),
	      const long s = -1) const {
      if (!(flags & EXRFLAG_INPUT))
	std::runtime_error("FileBase::read: Input file required for read");
      if (window.isEmpty())
	window = bufferWindow;
      _checkSample("read", window, s);
      const ChannelBuffer& channel = getChannel(channelName);
      if ((flags & EXRFLAG_DEEP) &&
	  !(channel.flags & EXRFLAG_SAMPLE_COUNT)) {
	unsigned int minNSample = minSampleCount(window);
	unsigned int maxNSample = maxSampleCount(window);
	if (minNSample != maxNSample && s >= minNSample) {
	  // Prevent reading in locations where there are missing samples
	  // TODO: Allow samples to be added?
	  Imath::V2i windowSize = _windowSize(window);
	  for (long y = 0; y < windowSize.y; y++) {
	    for (long x = 0; x < windowSize.x; x++) {
	      if (s < getSampleCount(y, x))
		data[y * windowSize.x + x] = channel.getData<T>(y + window.min.y,
								x + window.min.x,
								s);
	    }
	  }
	  return;
	}
      }
      getChannel(channelName).copyData(data, window, s);
    }
    /**
     * @brief Read data for a single tile from the file.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to copy data from.
     * @param[in] data Preallocated buffer that channel data should be
     *   copied into.
     * @param[in] i Tile index in the x direction.
     * @param[in] j Tile index in the y direction.
     * @param[in] s Sample to copy data for in a deep file. Must be
     *   provided for deep files and cannot be provided for scanline
     *   files.
     */
    template<typename T>
    void readTile(const std::string& channelName, T* data,
		  const long i, const long j,
		  const long s = -1) const {
      if (!(flags & EXRFLAG_TILED))
	std::runtime_error("FileBase::readTile: File is not tiled");
      Imath::Box2i window;
      window.min.x = dataWindow.min.x + i * header().tileDescription().xSize;
      window.max.x = window.min.x + header().tileDescription().xSize - 1;
      window.min.y = dataWindow.min.y + j * header().tileDescription().ySize;
      window.max.y = window.min.y + header().tileDescription().ySize - 1;
      read(channelName, data, window, s);
    }
    /**
     * @brief Read data for a single line from the file.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to copy data from.
     * @param[in] data Preallocated buffer that channel data should be
     *   copied into.
     * @param[in] y Line index in the y direction.
     * @param[in] s Sample to copy data for in a deep file. Must be
     *   provided for deep files and cannot be provided for scanline
     *   files.
     */
    template<typename T>
    void readLine(const std::string& channelName, T* data,
		  const long y, const long s = -1) const {
      if (flags & EXRFLAG_TILED)
	std::runtime_error("FileBase::readLine: File is tiled");
      Imath::Box2i window;
      window.min.x = dataWindow.min.x;
      window.max.x = window.min.x + dataWidth();
      window.min.y = dataWindow.min.y + y;
      window.max.y = window.min.y;
      read(channelName, data, window, s);
    }
    /**
     * @brief Read data for a single pixel from the file.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to copy data from.
     * @param[in] data Reference where pixel value should be read into.
     * @param[in] y Pixel index in the y direction.
     * @param[in] x Pixel index in the x direction.
     * @param[in] s Sample to copy data for in a deep file. Must be
     *   provided for deep files and cannot be provided for scanline
     *   files.
     */
    template<typename T>
    void readPixel(const std::string& channelName, T& data,
		   const long y, const long x, const long s = -1) {
      Imath::Box2i window;
      window.min.x = x;
      window.max.x = x;
      window.min.y = y;
      window.max.y = y;
      read(channelName, &data, window, s);
    }
    /**
     * @brief Read data for a single pixel from the file.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to copy data from.
     * @param[in] y Pixel index in the y direction.
     * @param[in] x Pixel index in the x direction.
     * @param[in] s Sample to copy data for in a deep file. Must be
     *   provided for deep files and cannot be provided for scanline
     *   files.
     * @return Value for pixel
     */
    template<typename T>
    T readPixel(const std::string& channelName,
		const long y, const long x, const long s = -1) {
      T out = 0;
      readPixel(channelName, out, y, x, s);
      return out;
    }
    
    // WRITE
    /**
     * @brief Write data to the file.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to write data to.
     * @param[in] data Buffer containing data that should be written to
     *   the file.
     * @param[in] window Bounds of data that should be written.
     * @param[in] s Sample to write data for in a deep file. Must be
     *   provided for deep files and cannot be provided for scanline
     *   files.
     */
    template<typename T>
    void write(const std::string& channelName, const T* data,
	       Imath::Box2i window=Imath::Box2i(),
	       const long s = -1) {
      if (!(flags & EXRFLAG_OUTPUT))
	std::runtime_error("FileBase::write: Output file required for write");
      if (window.isEmpty())
	window = bufferWindow;
      _checkSample("write", window, s);
      ChannelBuffer& channel = getChannel(channelName);
      if ((flags & EXRFLAG_DEEP) &&
	  !(channel.flags & EXRFLAG_SAMPLE_COUNT)) {
	unsigned int minNSample = minSampleCount(window);
	unsigned int maxNSample = maxSampleCount(window);
	if (minNSample != maxNSample && s >= minNSample) {
	  // Prevent writing in locations where there are missing samples
	  // TODO: Allow samples to be added?
	  Imath::V2i windowSize = _windowSize(window);
	  for (long y = 0; y < windowSize.y; y++) {
	    for (long x = 0; x < windowSize.x; x++) {
	      if (s < getSampleCount(y, x))
		channel.setData(data[y * windowSize.x + x],
				y + window.min.y,
				x + window.min.x,
				s);
	    }
	  }
	  return;
	}
      }
      getChannel(channelName).setData(data, window, s);
    }
    /**
     * @brief Write data for a single tile to the file.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to write data to.
     * @param[in] data Buffer containing data that should be written to
     *   the file.
     * @param[in] i Tile index in the x direction.
     * @param[in] j Tile index in the y direction.
     * @param[in] s Sample to write data for in a deep file. Must be
     *   provided for deep files and cannot be provided for scanline
     *   files.
     */
    template<typename T>
    void writeTile(const std::string& channelName, const T* data,
		   const long i, const long j,
		   const long s = -1) {
      if (!(flags & EXRFLAG_TILED))
	std::runtime_error("FileBase::writeTile: File is not tiled");
      Imath::Box2i window;
      window.min.x = dataWindow.min.x + i * header().tileDescription().xSize;
      window.max.x = window.min.x + header().tileDescription().xSize - 1;
      window.min.y = dataWindow.min.y + j * header().tileDescription().ySize;
      window.max.y = window.min.y + header().tileDescription().ySize - 1;
      write(channelName, data, window, s);
    }
    /**
     * @brief Write data for a single line to the file.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to write data to.
     * @param[in] data Buffer containing data that should be written to
     *   the file.
     * @param[in] y Line index in the y direction.
     * @param[in] s Sample to write data for in a deep file. Must be
     *   provided for deep files and cannot be provided for scanline
     *   files.
     */
    template<typename T>
    void writeLine(const std::string& channelName, const T* data,
		   const long y, const long s = -1) {
      if (flags & EXRFLAG_TILED)
	std::runtime_error("FileBase::writeLine: File is tiled");
      Imath::Box2i window;
      window.min.x = dataWindow.min.x;
      window.max.x = window.min.x + dataWidth();
      window.min.y = dataWindow.min.y + y;
      window.max.y = window.min.y;
      write(channelName, data, window, s);
    }
    /**
     * @brief Write data for a single pixel to the file.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to write data to.
     * @param[in] data Value to write.
     * @param[in] y Pixel index in the y direction.
     * @param[in] x Pixel index in the x direction.
     * @param[in] s Sample to write data for in a deep file. Must be
     *   provided for deep files and cannot be provided for scanline
     *   files.
     */
    template<typename T>
    void writePixel(const std::string& channelName, const T& data,
		    const long y, const long x, const long s = -1) {
      Imath::Box2i window;
      window.min.x = x;
      window.max.x = x;
      window.min.y = y;
      window.max.y = y;
      write(channelName, &data, window, s);
    }
    
    // CHUNK
    /**
     * @brief Get the first chunk in the current buffer. Incrementing
     *   the chunk will increment the buffer as necessary.
     * @param[in] chunkDim Size of the chunk in the x & y direcitons
     * @returns File chunk
     */
    CPPFileChunk chunk(Imath::V2i chunkDim=Imath::V2i(0, 0));

  };


  /**
   * @brief Container for tracking chunk info
   */
  class CPPFileChunk {
  public:
    FileBase* file;      //!< File containing chunk
    Imath::Box2i window; //!< Bounds of the chunk
    long x;              //!< Index of the chunk in the x direction
    long y;              //!< Index of the chunk in the y direction
    long xmax;           //!< Maximum index of chunks in the x direction
    long ymax;           //!< Maximum index of chunks in the y direction
    /**
     * @brief Empty constructor
     */
    CPPFileChunk() :
      file(nullptr), window(), x(-1), y(-1), xmax(-1), ymax(-1) {}
    /**
     * @brief Copy constructor
     * @param[in] Chunk to copy
     */
    CPPFileChunk(const CPPFileChunk& rhs) :
      file(rhs.file), window(rhs.window),
      x(rhs.x), y(rhs.y), xmax(rhs.xmax), ymax(rhs.ymax) {}
    /**
     * @brief Constructor
     * @param[in] fd File containing the chunk
     * @param[in] chunkDim Dimensions of the chunk. Required for tiles.
     */
    CPPFileChunk(FileBase* fd, Imath::V2i chunkDim=Imath::V2i(0, 0)) :
      file(fd), window(), x(0), y(0), xmax(0), ymax(0) {
      if (file->flags & EXRFLAG_TILED) {
	if (chunkDim.x == 0 || chunkDim.y == 0) {
	  chunkDim.x = file->header().tileDescription().xSize;
	  chunkDim.y = file->header().tileDescription().ySize;
	}
      } else {
	if (chunkDim.x == 0 || chunkDim.y == 0) {
	  chunkDim.x = 1;
	  chunkDim.y = file->dataWindow.max.y - file->dataWindow.min.y + 1;
	}
      }
      window.min = file->dataWindow.min;
      window.max.x = window.min.x + chunkDim.x - 1;
      window.max.y = window.min.y + chunkDim.y - 1;
      xmax = ((file->dataWindow.max.x - file->dataWindow.min.x + 1)/chunkDim.x) - 1;
      ymax = ((file->dataWindow.max.y - file->dataWindow.min.y + 1)/chunkDim.y) - 1;
    }
    /**
     * @brief Check if the chunk is deep
     * @returns true if the chunk is deep, false otherwise
     */
    bool isDeep() const {
      return file->isDeep();
    }
    /**
     * @brief Check if the chunk is inside the bounds of the data window
     * @returns true if the chunk is inside the data window, false
     *   otherwise
     */
    bool insideBounds() const {
      return (x <= xmax && y <= ymax);
    }
    /**
     * @brief Check if the entire file has been chunked.
     * @returns true if the file has been chunked, false otherwise.
     */
    bool isComplete() const {
      return (x > xmax || y > ymax);
    }
    /**
     * @brief Increment operator to go to next chunk in row-major order.
     * @returns Updated reference
     */
    CPPFileChunk& operator++() {
      if (x == xmax) {
	x = 0;
	y++;
      } else {
	x++;
      }
      _advance_window("CPPFileChunk::operator++", file->dataWindow, window,
		      false, true);
      if ((!isComplete()) &&
	  (!file->bufferWindow.intersects(window)))
	file->advanceFrameBuffer(false);
      return *this;
    }
    /**
     * @brief Decrement operator to go to previous chunk in row-major order.
     * @returns Updated reference
     */
    CPPFileChunk& operator--() {
      if (x == 0) {
	x = xmax;
	y--;
      } else {
	x--;
      }
      _advance_window("CPPFileChunk::operator--", file->dataWindow, window,
		      true, false);
      if ((!isComplete()) &&
	  (!file->bufferWindow.intersects(window)))
	file->advanceFrameBuffer(true);
      return *this;
    }

    // SAMPLE COUNT
    /**
     * @brief Get the maximum sample count within a window of the buffer
     * @returns Maximum sample count
     */
    unsigned int maxSampleCount() const {
      return file->maxSampleCount(window);
    }
    /**
     * @brief Get the minimum sample count within a window of the buffer
     * @returns Minimum sample count
     */
    unsigned int minSampleCount() const {
      return file->minSampleCount(window);
    }
    
    /**
     * @brief Read data from the file for this chunk.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to copy data from.
     * @param[in] data Preallocated buffer that channel data should be
     *   copied into.
     * @param[in] s Sample to copy data for in a deep file. Must be
     *   provided for deep files and cannot be provided for flat
     *   files.
     */
    template<typename T>
    void read(const std::string& channelName, T* data,
	      const long s = -1) const {
      file->read(channelName, data, window, s);
    }
    /**
     * @brief Write data to the file for this chunk.
     * @tparam T Type of data.
     * @param[in] channelName Name of channel to write data for.
     * @param[in] data Buffer containing data to be written for this chunk
     * @param[in] s Sample to write data for in a deep file. Must be
     *   provided for deep files and cannot be provided for flat
     *   files.
     */
    template<typename T>
    void write(const std::string& channelName, const T* data,
	       const long s = -1) {
      file->write(channelName, data, window, s);
    }
    
  };


CPPFileChunk FileBase::chunk(Imath::V2i chunkDim) {
  return CPPFileChunk(this, chunkDim);
}
  
#define SWITCH_OUTPUT_FILE(FILE_MACRO)					\
  if (flags & EXRFLAG_DEEP) {						\
    if (flags & EXRFLAG_TILED) {					\
      FILE_MACRO(Imf::DeepTiledOutputFile, Imf::DeepFrameBuffer);	\
    } else {								\
      FILE_MACRO(Imf::DeepScanLineOutputFile, Imf::DeepFrameBuffer);	\
    }									\
  } else {								\
    if (flags & EXRFLAG_TILED) {					\
      FILE_MACRO(Imf::TiledOutputFile, Imf::FrameBuffer);		\
    } else {								\
      FILE_MACRO(Imf::OutputFile, Imf::FrameBuffer);			\
    }									\
  }

  /**
   * @brief Output OpenEXR file class.
   */
  class CPPOutputFile : public FileBase {
  public:
    /**
     * @brief Constructor
     * @param[in] fname Full path to the file
     * @param[in] channelMap Mapping between channel names and
     *   enumerated types
     * @param[in] displayDim Dimensions of the image (in pixels)
     * @param[in] bufferDim Dimensions of the buffer window. If not
     *   provided, the buffer will cover the entire image.
     * @param[in] tileDim Dimensions of tiles within the image (in pixels)
     * @param[in] max_nsamples Maximum number of samples for an pixels
     *   in the file.
     * @param[in] compression Enumerated type of compression that should
     *   be used.
     * @param[in] levelMode Enumerated level mode that should be used.
     */
    CPPOutputFile(const std::string& fname,
		  const std::map<std::string, Imf::PixelType>& channelMap,
		  const Imath::V2i displayDim,
		  const Imath::V2i bufferDim=Imath::V2i(0,0),
		  const Imath::V2i tileDim=Imath::V2i(1,1),
		  const long max_nsamples = 1,
		  const Imf::Compression compression = Imf::NO_COMPRESSION,
		  const Imf::LevelMode levelMode = Imf::ONE_LEVEL) :
      FileBase(fname) {
      flags |= EXRFLAG_OUTPUT;
      // Create header
      displayWindow.min.setValue(0, 0);
      displayWindow.max.setValue(displayDim.x - 1, displayDim.y - 1);
      // TODO: Allow dataWindow to be passed?
      dataWindow.min.setValue(0, 0);
      dataWindow.max.setValue(displayDim.x - 1, displayDim.y - 1);
      Imf::Header header(displayWindow, dataWindow);
      header.compression() = compression;
      if (tileDim.x > 1 || tileDim.y > 1) {
	flags |= EXRFLAG_TILED;
	header.setTileDescription(
	  Imf::TileDescription(tileDim.x, tileDim.y, levelMode));
      }
      if (max_nsamples > 1) {
	flags |= EXRFLAG_DEEP;
      }
      if (flags & EXRFLAG_DEEP) {
	if (flags & EXRFLAG_TILED) {
	  header.setType(Imf::DEEPTILE);
	} else {
	  header.setType(Imf::DEEPSCANLINE);
	}
      }
      for (std::map<std::string, Imf::PixelType>::const_iterator it = channelMap.begin();
	   it != channelMap.end(); it++) {
	header.channels().insert(it->first, Imf::Channel(it->second));
      }
      // Create file
#define CREATE_FILE(T, TFB)			\
      createFile<T, TFB>(header, bufferDim)
      SWITCH_OUTPUT_FILE(CREATE_FILE);
#undef CREATE_FILE
    }
    /**
     * @brief Destroy the underlying OpenEXR file after writing out
     *   data in the buffer that has not already been written.
     */
    virtual ~CPPOutputFile() {
      flushBuffer();
#define DESTROY_FILE(T, TFB)			\
      destroyFile<T>()
      SWITCH_OUTPUT_FILE(DESTROY_FILE);
#undef DESTROY_FILE
    }

    /**
     * @brief Get the header for the file
     * @returns Header reference
     */
    const Imf::Header& header() const override {
      if (!file)
	throw std::runtime_error("header: File not initialized");
#define RETURN_HEADER(T, TFB)			\
      return ((T*)file)->header()
      SWITCH_OUTPUT_FILE(RETURN_HEADER);
#undef RETURN_HEADER
    }
    
    /**
     * @brief Create and assign a new frame buffer, first writing out
     *   the pre-existing buffer if it has not already been written.
     * @param[in] newBufferWindow Boundaries for the frame buffer that
     *   should be created.
     */
    void updateFrameBuffer(Imath::Box2i newBufferWindow=Imath::Box2i()) override {
      flushBuffer();
#define UPDATE_FRAME(T, TFB)				\
      _updateFrameBuffer<T, TFB>(newBufferWindow)
      SWITCH_OUTPUT_FILE(UPDATE_FRAME)
#undef UPDATE_FRAME
    }
    
    /**
     * @brief Create the underlying OpenEXR output file and initialize
     *   buffers that channel data can be assigned to
     * @tparam fileClass File class to create
     * @tparam fbClass Frame buffer class that should already exist
     * @param[in] bufferDim Dimensions of the new buffer (in pixels)
     */
    template<typename fileClass, typename fbClass>
    void createFile(Imf::Header& header, Imath::V2i bufferDim) {
      if (file)
	throw std::runtime_error("createFile: File already exists");
      addChannels(header);
      // if (max_nsamples > 1)
      // 	setSampleCountAll(max_nsamples);
      fileClass* tmp = new fileClass(filename.c_str(), header);
      file = (void*)tmp;
      updateFrameBuffer(initFrameBuffer(bufferDim));
    }
    
  private:
    /**
     * @brief Write out all of the tiles/scan lines coverred by the
     *   current buffer if the buffer is not empty and has not already
     *   been written.
     */
    void flushBuffer() {
      if (bufferWindow.isEmpty() || (flags & EXRFLAG_BUFFER_DONE))
	return;
      if (flags & EXRFLAG_TILED) {
	long xmin = bufferWindow.min.x / header().tileDescription().xSize;
	long xmax = bufferWindow.max.x / header().tileDescription().xSize;
	long ymin = bufferWindow.min.y / header().tileDescription().ySize;
	long ymax = bufferWindow.max.y / header().tileDescription().ySize;
	// Allow user to pass tile level?
	if (flags & EXRFLAG_DEEP) {
	  ((Imf::DeepTiledOutputFile*)file)->writeTiles(xmin, xmax,
							ymin, ymax, 0, 0);
	} else {
	  ((Imf::TiledOutputFile*)file)->writeTiles(xmin, xmax,
						    ymin, ymax, 0, 0);
	}
      } else {
	long ymin = bufferWindow.min.y;
	long ymax = bufferWindow.max.y;
	if (flags & EXRFLAG_DEEP) {
	  ((Imf::DeepScanLineOutputFile*)file)->writePixels(ymax - ymin + 1);
	} else {
	  ((Imf::OutputFile*)file)->writePixels(ymax - ymin + 1);
	}
      }
      flags |= EXRFLAG_BUFFER_DONE;
    }
    
  };

#undef SWITCH_OUTPUT_FILE
    
#define SWITCH_INPUT_FILE(FILE_MACRO)					\
  if (flags & EXRFLAG_DEEP) {						\
    if (flags & EXRFLAG_TILED) {					\
      FILE_MACRO(Imf::DeepTiledInputFile, Imf::DeepFrameBuffer);	\
    } else {								\
      FILE_MACRO(Imf::DeepScanLineInputFile, Imf::DeepFrameBuffer);	\
    }									\
  } else {								\
    if (flags & EXRFLAG_TILED) {					\
      FILE_MACRO(Imf::TiledInputFile, Imf::FrameBuffer);		\
    } else {								\
      FILE_MACRO(Imf::InputFile, Imf::FrameBuffer);			\
    }									\
  }

  /**
   * @brief Input OpenEXR file class.
   */
  class CPPInputFile : public FileBase {
  public:
    /**
     * @brief Constructor
     * @param[in] fname Full path to the file
     * @param[in] channelList List of channels that should be mapped.
     *   If not provided, all channels will be loaded.
     * @param[in] bufferDim Dimensions of the buffer window. If not
     *   provided, the buffer will cover the entire image.
     */
    CPPInputFile(const std::string& fname,
		 const std::vector<std::string>& channelList = {},
		 const Imath::V2i bufferDim=Imath::V2i(0,0)) :
      FileBase(fname) {
      flags |= EXRFLAG_INPUT;
      if (Imf::isTiledOpenExrFile(fname.c_str()))
	flags |= EXRFLAG_TILED;
      if (Imf::isDeepOpenExrFile(fname.c_str()))
	flags |= EXRFLAG_DEEP;
      if (Imf::isMultiPartOpenExrFile(fname.c_str()))
	flags |= EXRFLAG_MULTI;
#define CREATE_FILE(T, TFB)			\
      createFile<T, TFB>(channelList, bufferDim)
      SWITCH_INPUT_FILE(CREATE_FILE);
#undef CREATE_FILE
    }
    /**
     * @brief Destroy the underlying OpenEXR file
     */
    virtual ~CPPInputFile() {
#define DESTROY_FILE(T, TFB)			\
      destroyFile<T>()
      SWITCH_INPUT_FILE(DESTROY_FILE);
#undef DESTROY_FILE
    }

    /**
     * @brief Get the header for the file
     * @returns Header reference
     */
    const Imf::Header& header() const override {
      if (!file)
	throw std::runtime_error("header: File not initialized");
#define RETURN_HEADER(T, TFB)			\
      return ((T*)file)->header()
      SWITCH_INPUT_FILE(RETURN_HEADER);
#undef RETURN_HEADER
    }
    
    /**
     * @brief Create and assign a new frame buffer, loading data into it
     *   from the file.
     * @param[in] newBufferWindow Boundaries for the frame buffer that
     *   should be created.
     */
    void updateFrameBuffer(Imath::Box2i newBufferWindow=Imath::Box2i()) override {
#define UPDATE_FRAME(T, TFB)				\
      _updateFrameBuffer<T, TFB>(newBufferWindow);	\
      if (flags & EXRFLAG_DEEP) {			\
	_readPixelSampleCounts<T>();			\
      }
      SWITCH_INPUT_FILE(UPDATE_FRAME);
#undef UPDATE_FRAME
      loadBuffer();
    }
    
    /**
     * @brief Create the underlying OpenEXR input file and load data from
     *   the file to initialize the buffers.
     * @tparam fileClass File class to create
     * @tparam fbClass Frame buffer class that should already exist
     * @param[in] channelList List of channels that should be mapped.
     *   If not provided, all channels will be loaded.
     * @param[in] bufferDim Dimensions of the new buffer (in pixels)
     */
    template<typename fileClass, typename fbClass>
    void createFile(const std::vector<std::string>& channelList,
		    Imath::V2i bufferDim) {
      if (file)
	throw std::runtime_error("createFile: File already exists");
      fileClass* tmp = new fileClass(filename.c_str());
      file = (void*)tmp;
      displayWindow = tmp->header().displayWindow();
      dataWindow = tmp->header().dataWindow();
      addChannels(tmp->header(), channelList);
      updateFrameBuffer(initFrameBuffer(bufferDim));
    }
    
  private:
    /**
     * @brief Load all of the tiles/scan lines covered by the current
     *   buffer into the buffer if the buffer has not already been filled.
     */
    void loadBuffer() {
      if (flags & EXRFLAG_BUFFER_DONE)
	return;
      if (flags & EXRFLAG_TILED) {
	long xmin = bufferWindow.min.x / header().tileDescription().xSize;
	long xmax = bufferWindow.max.x / header().tileDescription().xSize;
	long ymin = bufferWindow.min.y / header().tileDescription().ySize;
	long ymax = bufferWindow.max.y / header().tileDescription().ySize;
	// Allow user to pass tile level?
	if (flags & EXRFLAG_DEEP) {
	  ((Imf::DeepTiledInputFile*)file)->readTiles(xmin, xmax, ymin, ymax, 0, 0);
	} else {
	  ((Imf::TiledInputFile*)file)->readTiles(xmin, xmax, ymin, ymax, 0, 0);
	}
      } else {
	long ymin = bufferWindow.min.y;
	long ymax = bufferWindow.max.y;
	if (flags & EXRFLAG_DEEP) {
	  ((Imf::DeepScanLineInputFile*)file)->readPixels(ymin, ymax);
	} else {
	  ((Imf::InputFile*)file)->readPixels(ymin, ymax);
	}
      }
      flags |= EXRFLAG_BUFFER_DONE;
    }

    /**
     * @brief Raise an error for an unsupprted input file class.
     */
    template<typename fileClass>
    void _readPixelSampleCounts() {
      throw std::runtime_error("_readPixelSampleCounts: Unsupported class");
    }
    /**
     * @brief Helper method to read in the sample counts for all of the
     *   tiles in the current buffer.
     */
    template<>
    void _readPixelSampleCounts<Imf::DeepTiledInputFile>() {
      Imf::DeepTiledInputFile* tmp = (Imf::DeepTiledInputFile*)file;
      long xmin = bufferWindow.min.x / header().tileDescription().xSize;
      long xmax = bufferWindow.max.x / header().tileDescription().xSize;
      long ymin = bufferWindow.min.y / header().tileDescription().ySize;
      long ymax = bufferWindow.max.y / header().tileDescription().ySize;
      tmp->readPixelSampleCounts(xmin, xmax, ymin, ymax);
      unsigned int* N = getSampleCount();
      setSampleCount(N, bufferWindow, true);
    }
    /**
     * @brief Helper method to read in the sample counts for all of the
     *   scan lines in the current buffer.
     */
    template<>
    void _readPixelSampleCounts<Imf::DeepScanLineInputFile>() {
      Imf::DeepScanLineInputFile* tmp = (Imf::DeepScanLineInputFile*)file;
      long ymin = bufferWindow.min.y;
      long ymax = bufferWindow.max.y;
      tmp->readPixelSampleCounts(ymin, ymax);
    }
    
  };

#undef SWITCH_INPUT_FILE

#undef FILE_METHOD
#undef FILE_METHOD_NORET
#undef FILE_METHOD_BASE
#undef SWITCH_TYPE
#undef SWITCH_TYPE_NOERROR
  
}
