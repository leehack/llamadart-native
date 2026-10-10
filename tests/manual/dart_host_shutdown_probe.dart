// Manual Linux host-shutdown probe. Build dart_exit_probe_shim.cpp as described
// in dart_exit_probe.dart, then run with PROBE_STEP_DELAY_MS=300:
// dart tests/manual/dart_host_shutdown_probe.dart <shim> <bundle> <scenario> <model>
// cooperative-finalizer-slow-native verifies a native call is active before
// kill, all worker exits / finalizers complete, and normal C exit preserves
// the earlier host callback and buffered output. raw-control is the known
// Dart VM diagnostic; cooperative-control requires no native bundle.
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

// Generates as llamadart does: one guarded native step, then Dart work.
final class _ProbeAnchor implements Finalizable {}

final _anchors = <Finalizable>[];
final _finalizers = <NativeFinalizer>[];

void _generate((SendPort, String, String, String, int) a) {
  final (ready, shimPath, bundle, model, tokens) = a;
  final shim = DynamicLibrary.open(shimPath);
  shim.lookupFunction<_OpenLibN, _OpenLibD>('shim_open_library')(_cstr(bundle));
  final session = shim.lookupFunction<_OpenSesN, _OpenSesD>(
    'shim_open_session',
  )(_cstr(model), tokens);
  final anchor = _ProbeAnchor();
  final finalizer = NativeFinalizer(
    shim.lookup<NativeFunction<Void Function(Pointer<Void>)>>(
      'shim_close_session',
    ),
  );
  _anchors.add(anchor);
  _finalizers.add(finalizer);
  finalizer.attach(anchor, session);
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
    if (++steps == 3) ready.send(session.address);
  }
}

Future<void> main(List<String> args) async {
  final [shimPath, bundle, scenario, model] = args;
  final shim = DynamicLibrary.open(shimPath);
  final registerHost = shim.lookupFunction<Void Function(), void Function()>(
    'shim_register_host',
  );
  final bufferOutput = shim.lookupFunction<Void Function(), void Function()>(
    'shim_buffer_output',
  );
  final freeCount = shim.lookupFunction<Int32 Function(), int Function()>(
    'shim_free_count',
  );
  final activeCalls = shim.lookupFunction<Int32 Function(), int Function()>(
    'shim_active_calls',
  );
  registerHost(); // Registered before the native bundle; must still run later.
  final exited = <ReceivePort>[];
  final stopped = <Future<dynamic>>[];
  final isolates = <Isolate>[];
  Future<dynamic> spawnWorker<T>(
    void Function(T) entry,
    T arg,
    ReceivePort ready,
  ) async {
    final onExit = ReceivePort();
    exited.add(onExit);
    stopped.add(onExit.first);
    isolates.add(await Isolate.spawn(entry, arg, onExit: onExit.sendPort));
    final value = await ready.first;
    ready.close();
    return value;
  }

  int? sessionAddress;
  if (!scenario.contains('control')) {
    final ready = ReceivePort();
    sessionAddress = await spawnWorker(_generate, (
      ready.sendPort,
      shimPath,
      bundle,
      model,
      scenario.contains('long') ? 1500 : 8,
    ), ready) as int;
  }
  final spinning = ReceivePort();
  await spawnWorker(_spin, spinning.sendPort, spinning);
  if (!scenario.startsWith('raw')) {
    if (scenario.contains('slow-native')) {
      final deadline = DateTime.now().add(const Duration(seconds: 3));
      while (activeCalls() == 0 && DateTime.now().isBefore(deadline)) {
        await Future<void>.delayed(const Duration(milliseconds: 1));
      }
      if (activeCalls() == 0) throw StateError('slow native call not active');
      stdout.writeln('NATIVE_ACTIVE_BEFORE_KILL');
    }
    final watch = Stopwatch()..start();
    for (final isolate in isolates) {
      isolate.kill(priority: Isolate.immediate);
    }
    await Future.wait(stopped).timeout(const Duration(seconds: 20));
    for (final port in exited) {
      port.close();
    }
    stdout.writeln('ALL_WORKERS_EXITED ${watch.elapsedMicroseconds}');
    stdout.writeln('NATIVE_CALLS_AFTER_EXIT ${activeCalls()}');
    if (activeCalls() != 0)
      throw StateError('native call survived worker exit');
    stdout.writeln('NATIVE_FREE_COUNT_AFTER_EXIT ${freeCount()}');
    if (sessionAddress != null && freeCount() != 1)
      throw StateError('native finalizer not completed before onExit');
  }
  bufferOutput();
  _cExit();
}
