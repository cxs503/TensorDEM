/* No-op provider of std::ios_base_library_init() under the GLIBCXX_3.4.32
 * version node.  The system libstdc++ (6.0.30, Loongnix 23) predates the
 * symbol; torch 2.4.0 (loongarch, built with GCC 13.2) requires it at load
 * time.  On this platform the standard streams are eagerly initialized by
 * libstdc++ static constructors, so an empty body is functionally equivalent
 * for torch's usage. */
void _ZSt21ios_base_library_initv(void)
{
}
