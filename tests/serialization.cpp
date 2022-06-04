#include <boost/ut.hpp>
#include <dllib/serialization.hpp>

namespace ut = boost::ut;

static ut::suite serialization_tests = [] {
  using namespace ut;
  using namespace dllib;

  "simple"_test = [] {
    std::stringstream ss;
    int v1 = 123;
    float v2 = 1.23;
    double v3 = 1.234;
    char v4 = 'a';
    long double v5 = 1.2345;
    unsigned long long v6 = 1'234'567;
    {
      int v1_local = v1;
      float v2_local = v2;
      double v3_local = v3;
      char v4_local = v4;
      long double v5_local = v5;
      unsigned long long v6_local = v6;

      Dump(ss, v1_local);
      Dump(ss, v2_local);
      Dump(ss, v3_local);
      Dump(ss, v4_local);
      Dump(ss, v5_local);
      Dump(ss, v6_local);
    }
    {
      int v1_local;
      float v2_local;
      double v3_local;
      char v4_local;
      long double v5_local;
      unsigned long long v6_local;

      Load(ss, v1_local);
      Load(ss, v2_local);
      Load(ss, v3_local);
      Load(ss, v4_local);
      Load(ss, v5_local);
      Load(ss, v6_local);

      expect(eq(v1_local, v1));
      expect(eq(v2_local, v2));
      expect(eq(v3_local, v3));
      expect(eq(v4_local, v4));
      expect(eq(v5_local, v5));
      expect(eq(v6_local, v6));
    }
  };

  "tensor"_test = [] {
    TTensor<int, 2, 3> t1 = {{1, 2, 0}, {3, 4, 5}};
    TTensor<int> t2 = 3;
    TTensor<float, 4, 3> t3 = {
      {1. /  1, 1. /  2, 1. /  3},
      {1. /  4, 1. /  5, 1. /  6},
      {1. /  7, 1. /  8, 1. /  9},
      {1. / 10, 1. / 11, 1. / 12},
    };

    std::stringstream ss;
    {
      TTensor<int, 2, 3> t1_local = t1;
      TTensor<int> t2_local = t2;
      TTensor<float, 4, 3> t3_local = t3;
      Dump(ss, t1_local);
      Dump(ss, t2_local);
      Dump(ss, t3_local);
    }
    {
      TTensor<int, 2, 3> t1_local;
      TTensor<int> t2_local;
      TTensor<float, 4, 3> t3_local;

      Load(ss, t1_local);
      Load(ss, t2_local);
      Load(ss, t3_local);

      expect(eq(t1, t1_local));
      expect(eq(t2, t2_local));
      expect(eq(t3, t3_local));
    }
  };

  "user_defined_type"_test = [] {
    struct TTest {
      int x;
      double y;
      TTensor<int, 2, 3> t;
      int five;

      std::tuple<int&, double&, TTensor<int, 2, 3>&> GetSerializationFields() const {
        return {
          const_cast<int&>(x),
          const_cast<double&>(y),
          const_cast<TTensor<int, 2, 3>&>(t),
        };
      }
    };

    std::stringstream ss;
    {
      TTest t{2, 3.3, {{1, 2, 3}, {4, 5, 6}}, 5};
      Dump(ss, t);
    }
    {
      TTest t{};
      expect(eq(t.five, 0));
      Load(ss, t);
      expect(eq(t.x, 2) &&
             eq(t.y, 3.3) &&
             eq(t.t, TTensor<int, 2, 3>{{1, 2, 3}, {4, 5, 6}}) &&
             eq(t.five, 0));
    }
  };
};
