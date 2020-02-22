#include <iostream>
#include <string.h>

using namespace std;

template<typename Key, typename Value>
class Cache {
    Value **cache;
    size_t cache_size;
public:
    Cache(size_t cache_size) {
        this->cache_size = cache_size;
        size_t size = cache_size * sizeof(Value *);
        this->cache = (Value **) malloc(size);
        memset(this->cache, 0, size);
    }

    Value *get(Key k) {
        int i;
        for (i = 0; i < this->cache_size; i++) {
            if (this->cache[i] == 0 || this->cache[i]->matches(k)) break;
        }

        // cache is full and didn't match
        if (i == this->cache_size) {
            i -= 1;
            delete this->cache[i];
            this->cache[i] = 0;
        }

        // if needed allocate else move item closer to beginning of the cache
        if (this->cache[i] == 0) {
            this->cache[i] = new Value(k);
        } else {
            if (i > 0) {
                swap(this->cache[i - 1], this->cache[i]);
                i -= 1;
            }
        }

        return this->cache[i];
    }

    void free(){
        for (int i = 0; i < this->cache_size; i++) {
            if (this->cache[i] != 0){
                delete this->cache[i];
                this->cache[i] = 0;
            }
        }
    }

    ~Cache(){
        cout << "Cache clear" << endl;
        this->free();
        ::free((void*)this->cache);
    }
};

class SimpleKey {
public:
    int x;
    int y;

    SimpleKey(int _x, int _y) : x(_x), y(_y) {}

    bool operator==(const SimpleKey &b) const {
        return this->x == b.x && this->y == b.y;
    }
};

class SimpleValue {
public:
    SimpleKey k;
    int z;

    SimpleValue(SimpleKey _k) : k(_k), z(_k.x + _k.y){}

    bool matches(const SimpleKey &b)const{
        return this->k == b;
    }

    ~SimpleValue(){
        cout << "Freeing " << z << endl;
    }
};

void f(Cache<SimpleKey, SimpleValue>& cache){
    SimpleValue* v = cache.get({1, 1});

    cout << cache.get(SimpleKey(1, 1))->z << endl;
    v->z = -1;
    cout << cache.get(SimpleKey(1, 1))->z << endl;
}

int main() {
    Cache<SimpleKey, SimpleValue> cache(3);

    cache.get(SimpleKey(0, 0));
    cache.get(SimpleKey(0, 1));

    f(cache);

    cache.get(SimpleKey(1, 2));
    cache.get(SimpleKey(1, 3));
    cout << cache.get(SimpleKey(1, 1))->z << endl;

    cache.free();
}