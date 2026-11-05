// ignore_for_file: deprecated_member_use
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:ffi/ffi.dart';
import 'package:llamadart/llamadart.dart';
import 'package:llamadart/src/backends/llama_cpp/llama_cpp_service.dart';

import 'passive_modules.dart';

String digest(String value) => sha256.convert(utf8.encode(value)).toString();

Map<String, Object?> queryProbe(List<Map<String, Object?>> modules) {
  final wrappers = modules
      .where(
        (m) => RegExp(
          r'[/\\](?:lib)?llamadart(?:-windows-x64)?\.dll$',
          caseSensitive: false,
        ).hasMatch(m['modulePath']! as String),
      )
      .toList();
  if (wrappers.length != 1)
    throw StateError('exactly one resident wrapper required');
  final wrapper = wrappers.single;
  final library = DynamicLibrary.open(wrapper['modulePath']! as String);
  const legacy = 'llama_dart_speculative_need_embd';
  const retained = 'llama_dart_speculative_need_embd_nextn';
  final hasNextn = library.providesSymbol(retained);
  if (!hasNextn) throw StateError('retained nextn export missing');
  final nextn = library
      .lookupFunction<
        Bool Function(Pointer<Void>),
        bool Function(Pointer<Void>)
      >(retained);
  final pointer = library.lookup<NativeFunction<Bool Function(Pointer<Void>)>>(
    retained,
  );
  final process = DynamicLibrary.process();
  final owner = process
      .lookupFunction<
        Int32 Function(Uint32, Pointer<Void>, Pointer<Pointer<Void>>),
        int Function(int, Pointer<Void>, Pointer<Pointer<Void>>)
      >('GetModuleHandleExW');
  final filename = process
      .lookupFunction<
        Uint32 Function(Pointer<Void>, Pointer<Utf16>, Uint32),
        int Function(Pointer<Void>, Pointer<Utf16>, int)
      >('GetModuleFileNameW');
  final handle = calloc<Pointer<Void>>();
  final name = calloc<Uint16>(32768).cast<Utf16>();
  try {
    if (owner(6, pointer.cast(), handle) == 0)
      throw StateError('nextn owner missing');
    final length = filename(handle.value, name, 32768);
    if (length == 0 || length >= 32768)
      throw StateError('nextn owner path missing');
    final ownerPath = name.toDartString(length: length);
    return {
      'wrapperPath': wrapper['modulePath'],
      'wrapperSha256': wrapper['sha256'],
      'hasLegacyQuery': library.providesSymbol(legacy),
      'hasNextn': hasNextn,
      'nextnNullFalse': !nextn(nullptr),
      'nextnOwnerPath': ownerPath,
      'nextnOwnerSha256': sha256
          .convert(File(ownerPath).readAsBytesSync())
          .toString(),
      'probeAfterPublicOutcome': true,
    };
  } finally {
    calloc.free(handle);
    calloc.free(name);
  }
}

Future<void> main(List<String> args) async {
  if (!Platform.isWindows || args.length != 5) {
    throw ArgumentError(
      'Windows required: MODE MODEL CACHE_ROOT MISSING_ROOT FAMILY',
    );
  }
  final mode = args[0];
  final expectedFamily = args[4];
  final engine = LlamaEngine(LlamaBackend());
  final result = <String, Object?>{
    'mode': mode,
    'family': expectedFamily,
    'maxTokens': 128,
    'threads': 2,
    'gpuLayers': 0,
    'completed': false,
    'disposed': false,
  };
  const prompt =
      'Repeat and continue this sequence exactly:\n'
      'alpha beta gamma delta epsilon zeta eta theta\n'
      'alpha beta gamma delta epsilon zeta eta theta\n'
      'alpha beta gamma delta epsilon zeta eta theta\n'
      'alpha beta gamma delta epsilon zeta eta theta\n'
      'alpha beta gamma delta epsilon zeta eta theta';
  result['fixture'] = {
    'strategy': 'ngram-simple',
    'sizeN': 4,
    'sizeM': 8,
    'minHits': 1,
    'promptSha256': digest(prompt),
  };
  const modelParams = ModelParams(
    contextSize: 2048,
    preferredBackend: GpuBackend.cpu,
    gpuLayers: 0,
    numberOfThreads: 2,
    numberOfThreadsBatch: 2,
  );
  Future<void> load() async {
    await engine.loadModel(args[1], modelParams: modelParams);
    if (await engine.getResolvedGpuLayers() != 0 ||
        !(await engine.getBackendName()).toLowerCase().contains('cpu')) {
      throw StateError('CPU zero-offload placement required');
    }
  }

  Future<String> generate(bool speculative) => engine
      .generate(
        prompt,
        params: GenerationParams(
          maxTokens: 128,
          temp: 0,
          topK: 1,
          topP: 1,
          minP: 0,
          penalty: 1,
          seed: 42,
          reusePromptPrefix: false,
          speculativeDecodingConfig: speculative
              ? const SpeculativeDecodingConfig(
                  strategy: SpeculativeDecodingStrategy.ngramSimple,
                  ngramSizeN: 4,
                  ngramSizeM: 8,
                  ngramMinHits: 1,
                )
              : null,
        ),
      )
      .join();
  try {
    await load();
    final baseline = await generate(false);
    final baselinePerf = await engine.getPerformanceContext();
    result['baselineEvalTokens'] = baselinePerf?.evalTokens;
    result['baselineOutput'] = baseline;
    result['baselineSha256'] = digest(baseline);
    result['baselineBytes'] = utf8.encode(baseline).length;
    await engine.unloadModel();
    await load();
    final speculative = await generate(true);
    final perf = await engine.getPerformanceContext();
    result['ngramOutput'] = speculative;
    result['ngramSha256'] = digest(speculative);
    result['drafted'] = perf?.speculativeDraftTokens;
    result['accepted'] = perf?.speculativeAcceptedDraftTokens;
    result['draftAttempts'] = perf?.speculativeDraftAttempts;
    result['ngramEvalTokens'] = perf?.evalTokens;
    result['fullOutputParity'] = baseline == speculative;
    if (baseline.isEmpty ||
        baseline != speculative ||
        (perf?.speculativeDraftTokens ?? 0) <= 0 ||
        (perf?.speculativeAcceptedDraftTokens ?? 0) <= 0) {
      throw StateError(
        'active ngram requires full parity and positive draft/accept counts',
      );
    }
    await engine.unloadModel();
    await load();
    final reloaded = await generate(false);
    final reloadPerf = await engine.getPerformanceContext();
    result['reloadEvalTokens'] = reloadPerf?.evalTokens;
    result['reloadOutput'] = reloaded;
    result['reloadParity'] = reloaded == baseline;
    if (result['reloadParity'] != true)
      throw StateError('reload parity failed');
    await engine.unloadModel();
    result['unloaded'] = true;
    result['completed'] = true;
  } catch (error) {
    result['errorType'] = error.runtimeType.toString();
    result['error'] = error.toString();
    exitCode = 1;
  } finally {
    try {
      await engine.dispose();
      result['disposed'] = true;
    } catch (error) {
      result['disposeError'] = error.toString();
      exitCode = 1;
    }
    try {
      final modules = await snapshot(
        result['completed'] == true
            ? 'after-stock-model-success'
            : 'after-stock-model-exception',
      );
      result['query'] = queryProbe(modules);
      await registryIdentity(
        modules,
        result['completed'] == true ? 'success' : 'exception',
      );
      String? discover(String root) =>
          LlamaCppService.resolveWindowsBackendModuleDirectory(
            resolvedExecutablePath: '$root/sdk/dart.exe',
            currentDirectoryPath: root,
            environment: const {},
          );
      result['resolverContexts'] = {
        for (final entry in {'mixed': args[2], 'missing': args[3]}.entries)
          entry.key: {
            'resolvedExecutablePath': '${entry.value}/sdk/dart.exe',
            'currentDirectoryPath': entry.value,
            'environment': <String, String>{},
          },
      };
      result['cacheSelected'] = discover(args[2]);
      result['missingLegacySelected'] = discover(args[3]);
    } catch (error) {
      result['observerError'] = error.toString();
      exitCode = 1;
    }
    stdout.writeln('CONSUMER_RESULT ${jsonEncode(result)}');
    await stdout.flush();
  }
}
