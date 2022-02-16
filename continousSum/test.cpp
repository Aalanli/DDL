#include <iostream>
#include <vector>
#include <torch/torch.h>

const float exp_overflow = 89;
const float exp_underflow = -104;


std::vector<int> calculate_over_under_flow(float theta, float stride, float beta) {
    int under = exp_overflow / beta + theta + stride;
    int over = exp_underflow / beta + stride;
    return {over, under};
}

int main()
{
    auto a = torch::tensor({1.0});
    auto b = torch::tensor({2.0});
    auto c = a.item<double>();
}