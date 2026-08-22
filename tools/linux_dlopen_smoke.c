#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s /path/to/libllamadart.so\n", argv[0]);
    return 2;
  }

  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (library == NULL) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }

  if (dlclose(library) != 0) {
    fprintf(stderr, "dlclose failed: %s\n", dlerror());
    return 1;
  }

  printf("Loaded %s\n", argv[1]);
  return 0;
}
