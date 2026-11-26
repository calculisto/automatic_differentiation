#include <calculisto/automatic_differentiation/automatic_differentiation.hpp>
    using calculisto::automatic_differentiation::dual_t;

    int
main ()
{
    // d (x²) / dx at x = 3.
        auto const
    x = dual_t <1> { 3., 0 };
        auto const
    y = x * x;
    return y.value == 9. and y.differentials[0] == 6. ? 0 : 1;
}
