#include <iostream>
#include <dllib/autograd.hpp>

void LogExample() {
  dllib::TVariable<dllib::TTensor<float, 2, 2>> var({{1, 2},
                                                     {3, 4}}, true);
  auto lg = dllib::Log(var);
  std::cout << lg->value << std::endl;
  dllib::Sum(lg)->Backward();
  std::cout << var->grad << std::endl;
}

void SqrtExample() {
  dllib::TVariable<dllib::TTensor<float, 2, 2>> var({{.01, 2},
                                                     {  3, 4}}, true);
  auto sqrt = dllib::Sqrt(var);
  std::cout << sqrt->value << std::endl;
  dllib::Sum(sqrt)->Backward();
  std::cout << var->grad << std::endl;
}

int main() {
  LogExample();
//  SqrtExample();
}
