// Manual probe, not run by CI: C exit() through FFI from the main isolate of
// a Dart program while its isolates use libllamadart. It shows what the exit
// handler of libllamadart costs a Dart VM, which aborts when its isolates run
// while exit() takes its time (docs/v060_2_wrapper_rebuild.md).
//
//   g++ -std=c++17 -shared -fPIC -Ithird_party/llama.cpp/include \
//     -Ithird_party/llama.cpp/ggml/include \
//     tests/manual/dart_exit_probe_shim.cpp -o libshim.so
//   dart tests/manual/dart_exit_probe.dart libshim.so <bundle-dir> \
//     <scenario> <model.gguf>
//
// <bundle-dir> holds libllamadart.so and the backend modules. A run is clean
// when it prints EXIT_PROBE_REACHED and exits with code 0. Append -slow-exit
// to a scenario for a host whose own exit handler takes 300 ms, or
// -late-slow-exit to a generating one for a handler that was registered after
// the one of libllamadart.
import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

typedef _OpenLibN = Int32 Function(Pointer<Uint8>);
typedef _OpenLibD = int Function(Pointer<Uint8>);
typedef _OpenSesN = Pointer<Void> Function(Pointer<Uint8>, Int32);
typedef _OpenSesD = Pointer<Void> Function(Pointer<Uint8>, int);
typedef _StepN = Int32 Function(Pointer<Void>);
typedef _StepD = int Function(Pointer<Void>);

final _malloc = DynamicLibrary.process()
    .lookupFunction<
      Pointer<Uint8> Function(IntPtr),
      Pointer<Uint8> Function(int)
    >('malloc');

Pointer<Uint8> _cstr(String s) {
  final bytes = utf8.encode(s);
  final p = _malloc(bytes.length + 1);
  for (var i = 0; i < bytes.length; i++) {
    p[i] = bytes[i];
  }
  p[bytes.length] = 0;
  return p;
}

Never _cExit() {
  stdout.writeln('EXIT_PROBE_REACHED');
  DynamicLibrary.process()
      .lookupFunction<Void Function(Int32), void Function(int)>('exit')(0);
  throw StateError('C exit returned');
}

void _spin(SendPort ready) {
  var rounds = 0;
  final bytes = utf8.encode('some text to decode, again and again');
  while (true) {
    if (utf8.decode(bytes).isEmpty) throw StateError('unreachable');
    if (++rounds == 100) ready.send(null);
  }
}

// An isolate that wakes every 5 ms and allocates a little, as an app with a
// heartbeat or an animation does.
void _tick(SendPort ready) {
  var ticks = 0;
  final bytes = utf8.encode('tick');
  Timer.periodic(const Duration(milliseconds: 5), (_) {
    final junk = List.generate(2000, (i) => utf8.decode(bytes));
    if (junk.isEmpty) throw StateError('unreachable');
    if (++ticks == 3) ready.send(null);
  });
}

// Generates as llamadart does: one guarded native step, then Dart work.
void _generate((SendPort, String, String, String, int) a) {
  final (ready, shimPath, bundle, model, tokens) = a;
  final shim = DynamicLibrary.open(shimPath);
  shim.lookupFunction<_OpenLibN, _OpenLibD>('shim_open_library')(_cstr(bundle));
  final session = shim.lookupFunction<_OpenSesN, _OpenSesD>(
    'shim_open_session',
  )(_cstr(model), tokens);
  final step = shim.lookupFunction<_StepN, _StepD>('shim_step');
  final bytes = utf8.encode('token text');
  var steps = 0;
  while (true) {
    final status = step(session);
    if (status != 0) {
      stderr.writeln('PROBE_STEP_STATUS $status');
    }
    for (var i = 0; i < 200; i++) {
      if (utf8.decode(bytes).isEmpty) throw StateError('unreachable');
    }
    if (++steps == 3) ready.send(null);
  }
}

// Runs four steps, reports, then stays idle: llamadart's worker after a
// generation that finished.
void _generateThenIdle((SendPort, String, String, String, int) a) {
  final (ready, shimPath, bundle, model, tokens) = a;
  final shim = DynamicLibrary.open(shimPath);
  shim.lookupFunction<_OpenLibN, _OpenLibD>('shim_open_library')(_cstr(bundle));
  final session = shim.lookupFunction<_OpenSesN, _OpenSesD>(
    'shim_open_session',
  )(_cstr(model), tokens);
  final step = shim.lookupFunction<_StepN, _StepD>('shim_step');
  for (var i = 0; i < 4; i++) {
    step(session);
  }
  final keepAlive = ReceivePort();
  ready.send(keepAlive.sendPort);
}

Future<void> main(List<String> args) async {
  final [shimPath, bundle, requested, model] = args;
  final shim = DynamicLibrary.open(shimPath);
  final openLib = shim.lookupFunction<_OpenLibN, _OpenLibD>(
    'shim_open_library',
  );
  final openSession = shim.lookupFunction<_OpenSesN, _OpenSesD>(
    'shim_open_session',
  );
  final step = shim.lookupFunction<_StepN, _StepD>('shim_step');
  final sleepingAtexit = shim
      .lookupFunction<Void Function(Int32), void Function(int)>(
        'shim_sleeping_atexit',
      );
  Future<void> spin() async {
    final ready = ReceivePort();
    await Isolate.spawn(_spin, ready.sendPort);
    await ready.first;
  }

  // A scenario with -slow-exit appended has an exit handler that takes 300 ms.
  // It is registered before libllamadart is opened, and so runs after the
  // handler of libllamadart.
  // With -late-slow-exit the handler is registered right before the exit
  // instead, and so runs before the handler of libllamadart.
  const slowExit = '-slow-exit';
  const lateSlowExit = '-late-slow-exit';
  final late = requested.endsWith(lateSlowExit);
  final scenario = late
      ? requested.substring(0, requested.length - lateSlowExit.length)
      : requested.endsWith(slowExit)
      ? requested.substring(0, requested.length - slowExit.length)
      : requested;
  if (!late && scenario != requested) sleepingAtexit(300);

  switch (scenario) {
    case 'control-spin': // no native library at all
      await spin();
      _cExit();
    case 'idle-spin': // model idle for longer than the settle time
      openLib(_cstr(bundle));
      final s = openSession(_cstr(model), 8);
      step(s);
      await spin();
      sleep(const Duration(milliseconds: 400));
      _cExit();
    case 'settle-spin': // exit right after a guarded call ended
      openLib(_cstr(bundle));
      final s = openSession(_cstr(model), 8);
      await spin();
      step(s);
      _cExit();
    case 'generating': // worker isolate alternates guarded calls and Dart
    case 'generating-spin':
    case 'long-decode-spin':
      final ready = ReceivePort();
      await Isolate.spawn(_generate, (
        ready.sendPort,
        shimPath,
        bundle,
        model,
        scenario == 'long-decode-spin' ? 1500 : 8,
      ));
      if (scenario == 'long-decode-spin') {
        await spin();
        sleep(const Duration(milliseconds: 300));
      } else {
        await ready.first;
        if (scenario == 'generating-spin') await spin();
      }
      if (late) sleepingAtexit(300);
      _cExit();
    case 'loaded-worker-idle': // llamadart's quit-loaded shape
    case 'loaded-worker-idle-timer': // the same with a periodic timer in main's group
    case 'loaded-worker-idle-spin':
      final ready = ReceivePort();
      await Isolate.spawn(_generateThenIdle, (
        ready.sendPort,
        shimPath,
        bundle,
        model,
        8,
      ));
      if (scenario == 'loaded-worker-idle-spin') await spin();
      if (scenario == 'loaded-worker-idle-timer') {
        final ticking = ReceivePort();
        await Isolate.spawn(_tick, ticking.sendPort);
        await ticking.first;
      }
      await ready.first;
      _cExit();
    case 'dart-io-exit-generating': // dart:io exit()
      final ready = ReceivePort();
      await Isolate.spawn(_generate, (
        ready.sendPort,
        shimPath,
        bundle,
        model,
        8,
      ));
      await ready.first;
      stdout.writeln('EXIT_PROBE_REACHED');
      exit(0);
    case 'return-generating': // main returns while a worker generates
      final ready = ReceivePort();
      final iso = await Isolate.spawn(_generate, (
        ready.sendPort,
        shimPath,
        bundle,
        model,
        8,
      ));
      await ready.first;
      stdout.writeln('EXIT_PROBE_REACHED');
      ready.close();
      iso.kill(priority: Isolate.immediate);
    default:
      throw ArgumentError.value(scenario);
  }
}
