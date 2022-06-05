#include <dllib/layer.hpp>
#include <dllib/optimizer.hpp>

#include <iostream>
#include <cassert>
#include <algorithm>

using namespace dllib;

int main() {

  auto create_gen = helpers::GetNormalGenerator<float>;
  constexpr size_t side = 6;
  TTensor<float, side, side> t;
  {
    auto& data = t.View<-1u>();
    std::generate(data.Begin(), data.End(), create_gen());
  }
  std::cout << "Goal: " << t << std::endl;
  FullyConnected<float, side, side> fc;
  auto opt = MakeOptimizerManager<TAdamOptimizerUnit>(.003);
  opt.AddParameter(fc);

  for (size_t i = 0; i < 10'000; ++i) {
    constexpr size_t n = 2;
    TTensor<float, n, side> inp{};
    {
      auto& data = inp.View<-1u>();
      std::generate(data.Begin(), data.End(), create_gen());
    }
    auto out = fc(DropOut(TVariable(inp, false)));
//    {
//      auto test = DropOut(fc(inp));
//      std::cout << test << std::endl << out->value << std::endl;
//      assert(test == out->value);
//    }
    {
      auto expected = MatrixProduct(inp, t);
      auto diff = TVariable(expected, false) - out;
      Sum(diff * diff)->Backward();
    }
//    std::cout << "Grad: " << get<0>(fc.GetParameters())->grad << std::endl;
    opt.Step();
    if ((i + 1) % 100 == 0) {
      dllib::CTensor auto& value = get<0>(fc.GetParameters())->value;
      std::cout << "Current: " << value << std::endl;
      std::cout << "Goal: " << t << std::endl;
      std::cout << "Difference: " << value - t << std::endl;
    }
  }
  std::cout << "Bias: " << get<0>(get<1>(fc.GetParameters()).GetParameters())->value << std::endl;
}
