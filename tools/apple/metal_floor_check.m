// Loads ggml's Metal backend from a libllamadart.dylib, compiling the embedded
// kernels as a user's device does.
//
// ggml compiles Metal source on the device, so the OS, not the Xcode that built
// the release, decides whether a kernel compiles. A runner's paravirtual device
// reports family Apple5 and no bfloat, so ggml leaves out the BF16 kernels there.
// --bf16 reports the two families that turn them on, as user devices do, and fails
// unless ggml then logs that it has bfloat.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dlfcn.h>
#import <objc/runtime.h>

static const NSInteger kFamilyMetal3 = 5001;
static BOOL (*originalSupportsFamily)(id, SEL, MTLGPUFamily);

static bool loggedBfloat = false;

static void logAndNoteBfloat(int level, const char * text, void * userData) {
    (void) level;
    (void) userData;
    fputs(text, stderr);
    if (strstr(text, "has bfloat") != NULL && strstr(text, "= true") != NULL) {
        loggedBfloat = true;
    }
}

static BOOL supportsFamilyWithBf16(id device, SEL selector, MTLGPUFamily family) {
    return family == MTLGPUFamilyApple6 || (NSInteger) family == kFamilyMetal3 ||
           originalSupportsFamily(device, selector, family);
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        const bool bf16 = argc == 3 && strcmp(argv[2], "--bf16") == 0;
        if (argc != 2 && !bf16) {
            fprintf(stderr, "usage: %s <libllamadart.dylib> [--bf16]\n", argv[0]);
            return 2;
        }
        void * library = dlopen(argv[1], RTLD_NOW);
        if (library == NULL) {
            fprintf(stderr, "%s\n", dlerror());
            return 2;
        }
        void * (*metal_reg)(void) = dlsym(library, "ggml_backend_metal_reg");
        void * (*reg_dev_get)(void *, size_t) = dlsym(library, "ggml_backend_reg_dev_get");
        void * (*dev_init)(void *, const char *) = dlsym(library, "ggml_backend_dev_init");
        void (*log_set)(void (*)(int, const char *, void *), void *) =
            dlsym(library, "ggml_log_set");
        if (metal_reg == NULL || reg_dev_get == NULL || dev_init == NULL || log_set == NULL) {
            fprintf(stderr, "the library does not export the ggml backend API\n");
            return 2;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "no Metal device, so this OS's compiler cannot be asked\n");
            return 2;
        }
        NSOperatingSystemVersion os = [[NSProcessInfo processInfo] operatingSystemVersion];
        printf("OS %ld.%ld.%ld, %s, BF16 %s\n", (long) os.majorVersion, (long) os.minorVersion,
               (long) os.patchVersion, device.name.UTF8String, bf16 ? "on" : "as reported");
        fflush(stdout);
        if (bf16) {
            Method method = class_getInstanceMethod(object_getClass(device),
                                                    @selector(supportsFamily:));
            originalSupportsFamily = (BOOL (*)(id, SEL, MTLGPUFamily)) method_setImplementation(
                method, (IMP) supportsFamilyWithBf16);
        }

        log_set(logAndNoteBfloat, NULL);
        void * metal_device = reg_dev_get(metal_reg(), 0);
        if (metal_device == NULL || dev_init(metal_device, NULL) == NULL) {
            fprintf(stderr, "the Metal backend failed to initialize\n");
            return 1;
        }
        if (bf16 && !loggedBfloat) {
            fprintf(stderr, "ggml did not turn on bfloat, so the BF16 kernels went unchecked\n");
            return 1;
        }
        printf("the Metal backend initialized\n");
        return 0;
    }
}
