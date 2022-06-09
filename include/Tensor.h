#pragma once

#include <vector>

class Tensor
{
public:
    enum class Type
    {
        Float,
        Half
    };

    Tensor(std::vector<int> shape, Type type, void *data, bool owning, int device);

    ~Tensor();

    // Tensor cannot be copy constructed nor assigned
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    Tensor(Tensor &&) = default;
    Tensor &operator=(Tensor &&) = default;

    bool isCuda() const { return m_device >= 0; }
    int device() const { return m_device; }
    Tensor cuda(int device) const;
    Tensor cpu() const;

    Type type() const { return m_type; }

    const std::vector<int> &shape() const { return m_shape; }
    // int shape(int dim) const { return m_shape[dim]; }
    int dimensions() const { return m_shape.size(); }

    template <typename T>
    T *data() { return reinterpret_cast<T *>(m_data); }
    template <typename T>
    const T *data() const { return reinterpret_cast<const T *>(m_data); }

    size_t byteSize() const;
    bool isOwning() const { return m_owning; }

private:
    Type m_type;
    void *m_data;
    bool m_owning;
    int m_device;

    std::vector<int> m_shape;
};