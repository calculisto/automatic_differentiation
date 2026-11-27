// The arithmetic: the operators, their values and derivatives, and compound
// assignment.
#include "test_tools.hpp"

TEST_CASE("Arithmetic")
{
SUBCASE("A dual with no differentials behaves normal number")
{
        const auto
    x = independent_variable <1> (3, 0);
        const auto
    p1 = 2 * x;
    CHECK(p1.value == 6);
    CHECK(p1.differentials[0] == 2);

        const auto
    a = dual_t <1> { 2 };
        const auto
    p2 = a * x;
    CHECK(p2.value == 6);
    CHECK(p2.differentials[0] == 2);
}
SUBCASE("Arithmetic, one value")
{
        const auto
    x = independent_variable <1> (2, 0);
    CHECK((+x).value == 2);
    CHECK((-x).value == -2);
    CHECK(x == 2);
    CHECK(2 == x);
    CHECK(x != 3);
    CHECK(3 != x);
    CHECK(x <  3);
    CHECK(1 <  x);
    CHECK(x >  1);
    CHECK(3 >  x);
    CHECK(x <= 3);
    CHECK(1 <= x);
    CHECK(x >= 1);
    CHECK(3 >= x);
    CHECK(x <= 2);
    CHECK(2 <= x);
    CHECK(x >= 2);
    CHECK(2 >= x);
    {
            const auto
        f = x + 1;
        CHECK(f.value == 3);
        CHECK(f.differentials[0] == 1);
    }
    {
            const auto
        f = 1 + x;
        CHECK(f.value == 3);
        CHECK(f.differentials[0] == 1);
    }
    {
            auto
        f = independent_variable <1> (1, 0);
        f += 1;
        CHECK(f.value == 2);
        CHECK(f.differentials[0] == 1);
    }
    {
            const auto
        f = x - 1;
        CHECK(f.value == 1);
        CHECK(f.differentials[0] == 1);
    }
    {
            const auto
        f = 1 - x;
        CHECK(f.value == -1);
        CHECK(f.differentials[0] == -1);
    }
    {
            auto
        f = independent_variable <1> (1, 0);
        f -= 1;
        CHECK(f.value == 0);
        CHECK(f.differentials[0] == 1);
    }
    {
            const auto
        f = x * 2;
        CHECK(f.value == 4);
        CHECK(f.differentials[0] == 2);
    }
    {
            const auto
        f = 2 * x;
        CHECK(f.value == 4);
        CHECK(f.differentials[0] == 2);
    }
    {
            const auto
        f = x / 2;
        CHECK(f.value == 1);
        CHECK(f.differentials[0] == .5);
    }
    {
            const auto
        f = 1 / x;
        CHECK(f.value == .5);
        CHECK(f.differentials[0] == -0.25);
    }
}
SUBCASE("Arithmetic, two values")
{
        const auto
    x = independent_variable <2> (2, 0);
        const auto
    y = independent_variable <2> (3, 1);
    CHECK(x != y);
    CHECK(x <  y);
    CHECK(y >  x);
    CHECK(x <= y);
    CHECK(y >= x);
    {
            const auto
        f = x + y;
        CHECK(f.value == 5);
        CHECK(f.differentials[0] == 1);
        CHECK(f.differentials[1] == 1);
    }
    {
            const auto
        f = x - y;
        CHECK(f.value == -1);
        CHECK(f.differentials[0] == 1);
        CHECK(f.differentials[1] == -1);
    }
    {
            const auto
        f = x * y;
        CHECK(f.value == 6);
        CHECK(f.differentials[0] == 3);
        CHECK(f.differentials[1] == 2);
    }
    {
            const auto
        f = x / y;
        CHECK(f.value == 2. / 3);
        CHECK(f.differentials[0] == 1. / 3);
        CHECK(f.differentials[1] == -2. / 9);
    }
}
SUBCASE("Polynomials")
{
        const auto
    x = independent_variable <3> (2, 0);
        const auto
    y = independent_variable <3> (3, 1);
        const auto
    z = independent_variable <3> (4, 2);
        const auto
    f = 3 * pow (x, 3) + 4 * pow (y, 2) + 5 * z + 3;
    CHECK(f.value == 83);
    CHECK(f.differentials[0] == 36);
    CHECK(f.differentials[1] == 24);
    CHECK(f.differentials[2] == 5);
}
SUBCASE("Unused dimensions")
{
        const auto
    x = independent_variable <3> (2, 0);
        const auto
    y = independent_variable <3> (3, 1);
        const auto
    z = 4.;
        const auto
    f = 3 * pow (x, 3) + 4 * pow (y, 2) + 5 * z + 3;
    CHECK(f.value == 83);
    CHECK(f.differentials[0] == 36);
    CHECK(f.differentials[1] == 24);
    CHECK(f.differentials[2] == 0);
}
    
}

TEST_CASE_TEMPLATE("Derivatives of the arithmetic"
    , Real, float, double, long double)
{
SUBCASE("Arithmetic")
{
    for (auto const& [x0, y0]: {
          std::pair { 3., 2. }
        , { -0.5, 4. }
        , { 2., -0.25 }
    })
    {
            const auto
        x = opaque <Real> (x0);
            const auto
        y = opaque <Real> (y0);
            using
        second_partials_t = std::array <std::array <Real, 2>, 2>;
        check_function ([] (auto const& a, auto const& b) { return a + b; }
            , std::array { x, y }
            , std::array <Real, 2> { 1, 1 }
            , second_partials_t {}
        );
        check_function ([] (auto const& a, auto const& b) { return a - b; }
            , std::array { x, y }
            , std::array <Real, 2> { 1, -1 }
            , second_partials_t {}
        );
        check_function ([] (auto const& a, auto const& b) { return a * b; }
            , std::array { x, y }
            , std::array { y, x }
            , second_partials_t {{ { 0, 1 }, { 1, 0 } }}
        );
        check_function ([] (auto const& a, auto const& b) { return a / b; }
            , std::array { x, y }
            , std::array { 1 / y, -x / (y * y) }
            , second_partials_t {{
                  { 0, -1 / (y * y) }
                , { -1 / (y * y), 2 * x / (y * y * y) }
              }}
        );
    }
}
}

TEST_CASE("Compound assignment")
{
    // a op= b is a op b, converted to the type of a, and returns a.
        const auto
    check = [] (auto a, auto const& b, auto operation, auto compound_assignment)
    {
            using
        value_t = typename decltype (a)::value_type;
            const auto
        expected = dual_cast <value_t> (operation (a, b));
            auto&
        r = compound_assignment (a, b);
        CHECK(&r == &a);
        CHECK(identical (a, expected));
    };
        const auto
    check_all = [&] (auto const& a, auto const& b)
    {
        check (a, b
            , [] (auto const& u, auto const& v) { return u + v; }
            , [] (auto& u, auto const& v) -> auto& { return u += v; }
        );
        check (a, b
            , [] (auto const& u, auto const& v) { return u - v; }
            , [] (auto& u, auto const& v) -> auto& { return u -= v; }
        );
        check (a, b
            , [] (auto const& u, auto const& v) { return u * v; }
            , [] (auto& u, auto const& v) -> auto& { return u *= v; }
        );
        check (a, b
            , [] (auto const& u, auto const& v) { return u / v; }
            , [] (auto& u, auto const& v) -> auto& { return u /= v; }
        );
    };
        const auto
    x = dual_t <2> { opaque <double> (1.5), { 2, -3 } };
        const auto
    y = dual_t <2> { opaque <double> (-0.7), { 0.25, 4 } };
        const auto
    x_float = dual_t <2, float> { opaque <float> (0.3), { -1, 0.5 } };
        using
    inner_t = dual_t <2>;
        const auto
    nested_x = dual_t <2, inner_t> {
          inner_t { opaque <double> (1.5), { 1, 0 } }
        , { inner_t { 1 }, inner_t { 0, { 0.5, 2 } } }
    };
        const auto
    nested_y = dual_t <2, inner_t> {
          inner_t { opaque <double> (-0.7), { 0, 1 } }
        , { inner_t { 3, { 1, 0 } }, inner_t { -2 } }
    };
    SUBCASE("With a dual")
    {
        check_all (x, y);
        check_all (x, x_float);
    }
    SUBCASE("With a dual of a wider type: converted to the type of a")
    {
        check_all (x_float, x);
    }
    SUBCASE("With a number")
    {
        check_all (x, opaque <int> (3));
        check_all (x, opaque <double> (0.75));
        check_all (x_float, opaque <double> (0.75));
    }
    SUBCASE("Nested duals")
    {
        check_all (nested_x, nested_y);
        check_all (nested_x, opaque <double> (0.75));
    }
    SUBCASE("With itself")
    {
            auto
        z = independent_variable <1> (opaque <double> (3), 0);
        z *= z;
        CHECK(z.value == 9);
        CHECK(z.differentials[0] == 6);
        z -= z;
        CHECK(z.value == 0);
        CHECK(z.differentials[0] == 0);
    }
}
