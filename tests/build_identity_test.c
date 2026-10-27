#include "llama_dart_build_identity.h"
#include "llama_dart_wrapper.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static const char * const kKeys[] = {
    "owner", "llama.cpp", "ggml", "patches", "config",
    "compiler", "target", "backends", "options",
};

static int is_value_char(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || strchr("._+,-", c) != NULL;
}

int main(void) {
    const char * identity = llama_dart_build_identity();
    assert(identity != NULL);
    // The library returns what this build stamped, not an older stamp.
    assert(strcmp(identity, LLAMA_DART_BUILD_IDENTITY) == 0);
    assert(llama_dart_build_identity() == identity);
    printf("build identity: %s\n", identity[0] ? identity : "(unidentified)");
    // Every lane that runs this test builds from git checkouts; an empty
    // identity there means the stamp silently stopped naming its inputs.
    assert(identity[0] != '\0');

    const char * prefix = "llamadart-native-build/1;";
    assert(strncmp(identity, prefix, strlen(prefix)) == 0);
    const char * cursor = identity + strlen(prefix);
    for (size_t i = 0; i < sizeof(kKeys) / sizeof(kKeys[0]); ++i) {
        const size_t key_length = strlen(kKeys[i]);
        assert(strncmp(cursor, kKeys[i], key_length) == 0);
        cursor += key_length;
        assert(*cursor == '=');
        ++cursor;
        const char * value = cursor;
        while (*cursor != '\0' && *cursor != ';') {
            assert(is_value_char(*cursor));
            ++cursor;
        }
        assert(cursor > value);
        if (i + 1 < sizeof(kKeys) / sizeof(kKeys[0])) {
            assert(*cursor == ';');
            ++cursor;
        }
    }
    assert(*cursor == '\0');
    return 0;
}
