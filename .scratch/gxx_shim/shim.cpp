// Shim providing std::ios_base_library_init() under the GLIBCXX_3.4.32
// version node, so torch 2.4.0 (built with GCC 13.2) loads on Loongnix 23
// whose system libstdc++ stops at GLIBCXX_3.4.30.
//
// std::ios_base_library_init() (added in GCC 13.2) explicitly initializes the
// standard iostream objects.  On a libstdc++ that still uses eager static
// initialization, touching the streams is the equivalent effect.
#include <iostream>

namespace std {
void ios_base_library_init()
{
    (void)std::cout;
    (void)std::cin;
    (void)std::cerr;
}
} // namespace std
