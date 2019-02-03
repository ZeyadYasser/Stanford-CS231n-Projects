#include <iostream>
#include "Gate.h"

int main()
{
    Gate* x = scaler(2);
    Gate* y = scaler(1);
    Gate* z = scaler(4);

    Gate* g0 = mul(x, x); // x^2
    Gate* g1 = add(g0, y); // x^2 + y
    Gate* g2 = mul(g1, z); // (x^2 + y) * z

    g0->Forward();
    g1->Forward();
    g2->Forward();

    g2->Backward(1);

    std::cout << "Gradient of x: " << x->Gradient() << std::endl; // 16
    std::cout << "Gradient of y: " << y->Gradient() << std::endl; // 4
    std::cout << "Gradient of z: " << z->Gradient() << std::endl; // 5

    return 0;
}
