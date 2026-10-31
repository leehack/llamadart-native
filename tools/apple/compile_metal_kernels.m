// Compiles the kernel sources of a directory with this OS's Metal compiler.
//
// ggml embeds Metal source and compiles it on the user's device, so the OS, not
// the Xcode that built the release, decides whether a kernel compiles. The macro
// sets are the ones ggml-metal-device.m passes on the devices of these OS versions.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s <kernel source directory>\n", argv[0]);
            return 2;
        }
        NSString * directory = [NSString stringWithUTF8String:argv[1]];
        NSOperatingSystemVersion os = [[NSProcessInfo processInfo] operatingSystemVersion];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "no Metal device, so this OS's compiler cannot be asked\n");
            return 2;
        }
        printf("OS %ld.%ld.%ld, %s\n", (long) os.majorVersion, (long) os.minorVersion,
               (long) os.patchVersion, device.name.UTF8String);

        NSArray<NSDictionary *> * macroSets = @[
            @{@"GGML_METAL_EMBED_LIBRARY": @"1", @"GGML_METAL_HAS_BF16": @"1"},
            @{@"GGML_METAL_EMBED_LIBRARY": @"1"},
        ];
        NSArray<NSString *> * names = [[[NSFileManager defaultManager]
            contentsOfDirectoryAtPath:directory error:nil]
            sortedArrayUsingSelector:@selector(compare:)];
        int compiled = 0;
        int failed = 0;
        for (NSString * name in names) {
            if (![name hasSuffix:@".metal"]) {
                continue;
            }
            NSString * source = [NSString
                stringWithContentsOfFile:[directory stringByAppendingPathComponent:name]
                                encoding:NSUTF8StringEncoding
                                   error:nil];
            for (NSDictionary * macros in macroSets) {
                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = macros;
                NSError * error = nil;
                compiled++;
                if ([device newLibraryWithSource:source options:options error:&error] == nil) {
                    failed++;
                    printf("FAILED %s (%s BF16)\n%s\n", name.UTF8String,
                           macros[@"GGML_METAL_HAS_BF16"] ? "with" : "without",
                           error.localizedDescription.UTF8String);
                }
            }
        }
        printf("%d of %d compiles failed\n", failed, compiled);
        return (compiled == 0 || failed != 0) ? 1 : 0;
    }
}
