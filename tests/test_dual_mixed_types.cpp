// Mixed types: duals of different floating-point types, duals and numbers,
// and nested duals, converted as numbers are.
#include "test_tools.hpp"

TEST_CASE("Mixed types")
{
        const auto
    dual_float = dual_t <2, float> { opaque <float> (0.7), { 2, -3 } };
        const auto
    dual_double = dual_t <2, double> { opaque <double> (1.3), { -0.5, 5 } };
        const auto
    dual_long_double = dual_t <2, long double> { opaque <long double> (0.4), { 1, 0.25 } };
        const auto
    float_number = opaque <float> (0.6);
        const auto
    double_number = opaque <double> (1.7);
        const auto
    long_double_number = opaque <long double> (0.9);
        const auto
    int_number = opaque <int> (2);
SUBCASE("dual_cast")
{
        const auto
    d = dual_cast <double> (dual_float);
    static_assert (std::same_as <decltype (d), dual_t <2, double> const>);
    CHECK(d.value == static_cast <double> (dual_float.value));
    CHECK(d.differentials[0] == 2);
    CHECK(d.differentials[1] == -3);
    static_assert (has_dual_cast <dual_t <2, float>, long double>);
    static_assert (not has_dual_cast <dual_t <2, float>, int>);
}
    const auto
check_all_pairs = [&] (auto f)
{
    check_mixed (f, dual_float, dual_double);
    check_mixed (f, dual_double, dual_float);
    check_mixed (f, dual_float, dual_long_double);
    check_mixed (f, dual_long_double, dual_double);
    check_mixed (f, dual_float, double_number);
    check_mixed (f, double_number, dual_float);
    check_mixed (f, dual_double, int_number);
    check_mixed (f, int_number, dual_double);
    check_mixed (f, dual_double, float_number);
    check_mixed (f, float_number, dual_double);
    check_mixed (f, dual_double, long_double_number);
    check_mixed (f, long_double_number, dual_double);
    check_mixed (f, dual_long_double, int_number);
    check_mixed (f, int_number, dual_long_double);
};
SUBCASE("Arithmetic")
{
    check_all_pairs ([] (auto const& a, auto const& b) { return a + b; });
    check_all_pairs ([] (auto const& a, auto const& b) { return a - b; });
    check_all_pairs ([] (auto const& a, auto const& b) { return a * b; });
    check_all_pairs ([] (auto const& a, auto const& b) { return a / b; });
}
SUBCASE("Functions of two arguments")
{
    check_all_pairs (FUNCTION (atan2));
    check_all_pairs (FUNCTION (hypot));
    check_all_pairs (FUNCTION (pow));
}
SUBCASE("Functions of two arguments of <cmath>, after pow")
{
        const auto
    remquo_value = [] (auto const&... arguments)
    {
            using std::remquo;
            int
        quotient;
        return remquo (arguments..., &quotient);
    };
    check_all_pairs (FUNCTION (fmod));
    check_all_pairs (FUNCTION (remainder));
    check_all_pairs (remquo_value);
    check_all_pairs (FUNCTION (copysign));
    check_all_pairs (FUNCTION (fdim));
    check_all_pairs (FUNCTION (fmax));
    check_all_pairs (FUNCTION (fmin));
}
SUBCASE("Comparison functions")
{
    // Of the values, converted as for the comparison operators.
    CHECK(isgreater (dual_double, dual_float) == std::isgreater (dual_double.value, dual_float.value));
    CHECK(isless (dual_float, double_number) == std::isless (dual_float.value, double_number));
    CHECK(islessequal (int_number, dual_long_double) == std::islessequal (int_number, dual_long_double.value));
}
SUBCASE("Functions of three arguments")
{
        const auto
    check_combinations = [&] (auto f)
    {
        check_mixed (f, dual_float, dual_double, dual_long_double);
        check_mixed (f, dual_float, double_number, int_number);
        check_mixed (f, int_number, dual_float, dual_double);
        check_mixed (f, float_number, double_number, dual_long_double);
    };
    check_combinations (FUNCTION (fma));
    check_combinations (FUNCTION (lerp));
    check_mixed (FUNCTION (hypot), dual_float, dual_double, dual_long_double);
    check_mixed (FUNCTION (hypot), dual_float, dual_double, float_number);
    check_mixed (FUNCTION (hypot), dual_float, int_number, dual_double);
    check_mixed (FUNCTION (hypot), int_number, dual_float, dual_long_double);
    check_mixed (FUNCTION (hypot), dual_float, double_number, int_number);
    check_mixed (FUNCTION (hypot), int_number, dual_float, double_number);
    check_mixed (FUNCTION (hypot), long_double_number, int_number, dual_float);
}
SUBCASE("Nested duals")
{
        using
    nested_float_t = dual_t <2, dual_t <2, float>>;
        using
    nested_double_t = dual_t <2, dual_t <2, double>>;
        const auto
    nested_float = nested_float_t {
          { opaque <float> (0.7), { 2, -3 } }
        , { dual_t <2, float> { 1, { 0.5, 0 } }, dual_t <2, float> { -2, { 0, 4 } } }
    };
        const auto
    nested_double = nested_double_t {
          { opaque <double> (1.3), { -0.5, 5 } }
        , { dual_t <2> { 3, { 0, -1 } }, dual_t <2> { 0.25, { 2, 0 } } }
    };
        const auto
    d = dual_cast <dual_t <2, double>> (nested_float);
    static_assert (std::same_as <decltype (d), nested_double_t const>);
    CHECK(d.value.value == static_cast <double> (nested_float.value.value));
    CHECK(d.differentials[1].differentials[1] == 4);
    static_assert (has_dual_cast <nested_float_t, dual_t <2, long double>>);
    static_assert (not has_dual_cast <nested_float_t, double>);
    static_assert (not has_dual_cast <nested_float_t, dual_t <1>>);
    static_assert (not has_dual_cast <dual_t <2, float>, dual_t <2>>);

    check_mixed ([] (auto const& a, auto const& b) { return a + b; }, nested_float, nested_double);
    check_mixed ([] (auto const& a, auto const& b) { return a * b; }, nested_double, nested_float);
    check_mixed ([] (auto const& a, auto const& b) { return a / b; }, nested_float, double_number);
    check_mixed ([] (auto const& a, auto const& b) { return a - b; }, int_number, nested_double);
    check_mixed (FUNCTION (atan2), nested_float, nested_double);
    check_mixed (FUNCTION (hypot), double_number, nested_float);
    check_mixed (FUNCTION (hypot), nested_float, int_number, nested_double);
    check_mixed (FUNCTION (pow), nested_float, nested_double);
    check_mixed (FUNCTION (pow), nested_float, int_number);
    check_mixed (FUNCTION (pow), double_number, nested_float);
    check_mixed (FUNCTION (fmod), nested_float, nested_double);
    check_mixed (FUNCTION (copysign), double_number, nested_float);
    check_mixed (FUNCTION (fmax), nested_float, int_number);
    check_mixed (FUNCTION (fma), nested_float, double_number, nested_double);
    check_mixed (FUNCTION (lerp), int_number, nested_double, nested_float);
}
SUBCASE("A dual of float and an int")
{
    // As with numbers, an int gives a float, here without the warning that
    // -Wconversion gives for a float and a non-constant int.
        const auto
    r = dual_float * int_number;
    static_assert (std::same_as <decltype (r), dual_t <2, float> const>);
    CHECK(r.value == dual_float.value * static_cast <float> (int_number));
    CHECK(r.differentials[0] == 2 * static_cast <float> (int_number));
}
SUBCASE("Compound assignment")
{
    // As for numbers, the result is converted to the type of the dual.
        auto
    d = dual_float;
    d += double_number;
    CHECK(d.value == static_cast <float> (dual_float.value + double_number));
    d -= double_number;
    CHECK(d.value == static_cast <float> (static_cast <float> (dual_float.value + double_number) - double_number));
}
}
