#if !defined NVTX_H
#define NVTX_H

#include <nvToolsExt.h>

namespace nvtx {

class Marker {
 public:
  enum Color { red, green, blue, yellow, black };

  Marker(const char* message, unsigned color = 0xff00ff00);
  Marker(const char* message, Marker::Color color);
  void start();
  void end();

 private:
  unsigned int convert(Color color);
  nvtxEventAttributes_t _attributes;
  nvtxRangeId_t _id;
};

}  // end namespace nvtx

#endif
