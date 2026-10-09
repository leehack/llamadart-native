// Linked into every shared library of the Linux bundle: libllamadart and the
// llama.cpp libraries next to it (CMakeLists.txt).
//
// The C++ runtime registers the destructor of each static of a library
// through __cxa_atexit. The linker binds those calls to this definition,
// which registers nothing, so the statics of the library are never destroyed.
//
// C exit() destroys statics while other threads are still inside llama.cpp,
// which then read freed memory (https://github.com/leehack/llamadart/issues/949).
// On Apple platforms llama.cpp is part of libllamadart, whose exit teardown
// runs before the first of those statics is destroyed. Here the libraries are
// separate, and exit handlers run in reverse order of registration: no handler
// of libllamadart precedes the destructor of a static that another library
// creates later, in the middle of a model load for one. The handler that
// libllamadart does register ends the process for a call in flight that it
// counts; a static that is never destroyed stays valid for the calls that it
// does not.
#if defined(__linux__) && !defined(__ANDROID__)
__attribute__((visibility("hidden"), used)) int
__cxa_atexit(void (*destroy)(void *), void *object, void *dso_handle) {
  (void)destroy;
  (void)object;
  (void)dso_handle;
  return 0;
}
#endif
