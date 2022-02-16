#include <iostream>
#include <vector>
#include <math.h>
#include <string>

template <typename T>
void print(std::vector<T> s) {
    for (T i:s) {
        std::cout << i << " ";
    }
    std::cout << "\n";
}

template <typename scalar_t>
inline scalar_t weight_kernel(scalar_t x, scalar_t a, scalar_t b) {
    return 1 / (1 + pow(abs(x / a), 2 * b));
}

template <typename scalar_t>
inline void d_weight_kernel(scalar_t x, scalar_t a, scalar_t b, scalar_t* w, scalar_t* dA, scalar_t* dB) {
    auto v = pow(abs(x / a), 2 * b);
    w[0] = 1 / (1 + v);
    auto vp = 2 / (1 / v + 2 + v);
    dA[0] = b * vp / a;
    dB[0] = -log(abs(x / a) + 1e-7) * vp;
} 

int main() {
    typedef float t;
    t w, da, db;
    for (int i=0; i < 50; i++) {
        d_weight_kernel<t>((t) i, 5.5, 5.5, &w, &da, &db);
        print<t>({(t) i, w, da, db});
    }
}