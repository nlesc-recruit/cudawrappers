#if !defined NVTX_H
#define NVTX_H

#if !defined(__HIP__)
#include <nvToolsExt.h>
#endif

namespace nvtx {

class Marker {
 public:
  enum Color { red, green, blue, yellow, black };

#if defined(__HIP__)
  explicit Marker(const char* message, unsigned color = Color::green) {}

#else
  explicit Marker(const char* message, unsigned color = Color::green)
      : _attributes{0} {
    _attributes.version = NVTX_VERSION;
    _attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    _attributes.colorType = NVTX_COLOR_ARGB;
    _attributes.color = color;
    _attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    _attributes.message.ascii = message;
  }
#endif

  Marker(const char* message, Color color) : Marker(message, convert(color)) {}

  void start() {
#if !defined(__HIP__)
    _id = nvtxRangeStartEx(&_attributes);
#endif
  }

  void end() {
#if !defined(__HIP__)
    nvtxRangeEnd(_id);
#endif
  }

 private:
  unsigned int convert(Color color) {
    switch (color) {
      case red:
        return 0xffff0000;
      case green:
        return 0xff00ff00;
      case blue:
        return 0xff0000ff;
      case yellow:
        return 0xffffff00;
      case black:
        return 0xff000000;
      default:
        return 0xff00ff00;
    }
  }

#if !defined(__HIP__)
  nvtxEventAttributes_t _attributes;
  nvtxRangeId_t _id;
#endif
};

}  // end namespace nvtx

#endif
