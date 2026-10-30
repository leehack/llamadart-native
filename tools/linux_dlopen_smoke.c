#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint32_t (*tts_model_types_fn)(void);

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "Usage: %s /path/to/libllamadart.so [expected-tts-model-types]\n", argv[0]);
    return 2;
  }

  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (library == NULL) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }

  if (argc == 3) {
    tts_model_types_fn supported = NULL;
    *(void **)&supported = dlsym(library, "llama_dart_tts_supported_model_types");
    if (supported == NULL) {
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
      return 1;
    }
    const uint32_t expected = (uint32_t)strtoul(argv[2], NULL, 0);
    const uint32_t actual = supported();
    if (actual != expected) {
      fprintf(stderr, "TTS model types 0x%x, expected 0x%x\n", actual, expected);
      return 1;
    }
  }

  if (dlclose(library) != 0) {
    fprintf(stderr, "dlclose failed: %s\n", dlerror());
    return 1;
  }

  printf("Loaded %s\n", argv[1]);
  return 0;
}
