#include "cache.h"

DeviceSizeKey::DeviceSizeKey(int d, int b, int w, int h) : device(d), batch(b), width(w), height(h) {}

bool DeviceSizeKey::operator==(const DeviceSizeKey &o) const {
    return this->device == o.device && this->batch == o.batch && this->width == o.width && this->height == o.height;
}

std::ostream &operator<<(std::ostream &os, DeviceSizeKey const &m) {
    return os << "(device: " << m.device << ", batch: " << m.batch << ", width: " << m.width << ", height: " << m.height
              << ")";
}