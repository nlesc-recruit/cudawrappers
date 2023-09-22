#if !defined NVTX_H
#define NVTX_H

#include <nvToolsExt.h>

namespace nvtx {

class Marker {
 public:
  enum Color { red, green, blue, yellow, black };

  explicit Marker(const char* message, unsigned color = Color::green)
      : _attributes{0} {
    _attributes.version = NVTX_VERSION;
    _attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    _attributes.colorType = NVTX_COLOR_ARGB;
    _attributes.color = color;
    _attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    _attributes.message.ascii = message;
  }

  Marker(const char* message, Color color) : Marker(message, convert(color)) {}

  void start() { _id = nvtxRangeStartEx(&_attributes); }

  void end() { nvtxRangeEnd(_id); }

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

  nvtxEventAttributes_t _attributes;
  nvtxRangeId_t _id;
};

}  // end namespace nvtx

#endif
