#include "cache.h"

DeviceSizeKey::DeviceSizeKey(int d, int b, int w, int h, int c, int p) : device(d), batch(b), width(w), height(h),
                                                                         channels(c), precision(p) {}

bool DeviceSizeKey::operator==(const DeviceSizeKey &o) const {
    return this->device == o.device && this->batch == o.batch && this->width == o.width && this->height == o.height &&
           this->channels == o.channels && this->precision == o.precision;
}

std::ostream &operator<<(std::ostream &os, DeviceSizeKey const &m) {
    return os << "(device: " << m.device << ", batch: " << m.batch << ", width: " << m.width << ", height: " << m.height
              << ", channels: " << m.channels << ", precision: " << m.precision << ")";
}