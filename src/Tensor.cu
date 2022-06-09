#include "Tensor.h"
#include "utils.h"

Tensor::Tensor(std::vector<int> shape, Type type, void *data, bool owning, int device)
    : m_shape(shape), m_type(type), m_data(data), m_owning(owning), m_device(device){}

Tensor::~Tensor(){
    if(m_owning){
        if(m_device == -1){
            free(m_data);
        }else{
            checkCudaErrors(cudaSetDevice(m_device));
            checkCudaErrors(cudaFree(m_data));
        }
    }
}

Tensor Tensor::cuda(int device) const{
    checkCudaErrors(cudaSetDevice(device));

    size_t size = byteSize();

    void* data;
    checkCudaErrors(cudaMalloc(&data, size));
    checkCudaErrors(cudaMemcpy(data, m_data, size, m_device == -1 ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice));

    return Tensor(m_shape, m_type, data, true, device);
}

Tensor Tensor::cpu() const{
    size_t size = byteSize();
    void* data = malloc(size);

    if(m_device == -1){
        memcpy(data, m_data, size);
    }else{
        checkCudaErrors(cudaSetDevice(m_device));
        checkCudaErrors(cudaMemcpy(data, m_data, size, cudaMemcpyDeviceToHost));
    }

    return Tensor(m_shape, m_type, data, true, -1);
}

size_t  Tensor::byteSize() const{
    size_t size = 1;
    for(auto x : m_shape)
        size *= x;
    
    switch(m_type){
        case Type::Float:
            size *= 4;
            break;
        case Type::Half:
            size *= 2;
            break;
    }

    return size;
}