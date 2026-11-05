param([Parameter(Mandatory=$true)][string]$Scratch,[Parameter(Mandatory=$true)][string]$CoreCheckout)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$H=$PSScriptRoot
$python=(Get-Command python).Source
$verify=Join-Path $H 'verify_evidence.py'
$spec=Get-Content (Join-Path $H 'inputs.json') -Raw | ConvertFrom-Json
$stage=Join-Path $Scratch 'stage';$out=Join-Path $Scratch 'out'
if(Test-Path -LiteralPath $Scratch){throw 'fresh scratch required; evidence must not be overwritten'}
New-Item -ItemType Directory $stage,$out | Out-Null
$cases=@();$childrenExited=$true;$completed=$false;$failure=$null
function Hash([string]$Path){(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()}
function SaveJson([string]$Path,$Value){$Value | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $Path -Encoding utf8}
function FileReceipt([string]$Path){@{path=[IO.Path]::GetFileName($Path);bytes=(Get-Item -LiteralPath $Path).Length;sha256=Hash $Path}}
function InvokeChild([string]$Name,[string]$Exe,[string[]]$Arguments,[string]$Cwd,[int]$Seconds=600){
  if($Name -cnotmatch '^[a-z0-9-]+$'){throw 'invalid child name'}
  $tmp=Join-Path $stage "tmp-$Name";$pub=Join-Path $stage "pub-$Name"
  if((Test-Path $tmp) -or (Test-Path $pub)){throw 'child state already exists; no retry'}
  New-Item -ItemType Directory $tmp,$pub | Out-Null
  $start=[Diagnostics.ProcessStartInfo]::new();$start.FileName=$Exe;$start.WorkingDirectory=$Cwd
  $start.UseShellExecute=$false;$start.CreateNoWindow=$true;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
  foreach($argument in $Arguments){$start.ArgumentList.Add($argument)}
  foreach($key in @('TEMP','TMP','TMPDIR')){$start.Environment[$key]=$tmp}
  $start.Environment['PUB_CACHE']=$pub
  SaveJson "$out/$Name-process.json" @{executable=$Exe;arguments=$Arguments;cwd=$Cwd;temp=$tmp;pubCache=$pub;fresh=$true;path=$env:Path;runtimeOverride=$env:LLAMADART_NATIVE_LIB_DIR;backendOverride=$env:LLAMADART_BACKEND_MODULE_DIR}
  $process=[Diagnostics.Process]::new();$process.StartInfo=$start;$timedOut=$false;$exit=-1;$started=$false
  try {
    if(-not $process.Start()){throw 'child did not start'}
    $started=$true
    $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync()
    if(-not $process.WaitForExit($Seconds*1000)){$timedOut=$true;$process.Kill($true);$process.WaitForExit()}
    $exit=$process.ExitCode
    [IO.File]::WriteAllText("$out/$Name.stdout.log",$stdout.GetAwaiter().GetResult(),[Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText("$out/$Name.stderr.log",$stderr.GetAwaiter().GetResult(),[Text.UTF8Encoding]::new($false))
    return @{name=$Name;exitCode=$exit;timedOut=$timedOut;childExited=$process.HasExited;workingDirectory=$start.WorkingDirectory;stdout=FileReceipt "$out/$Name.stdout.log";stderr=FileReceipt "$out/$Name.stderr.log"}
  } finally {
    if($started -and -not $process.HasExited){$process.Kill($true);$process.WaitForExit();$script:childrenExited=$process.HasExited}
    $process.Dispose()
  }
}
function Must([string]$Name,[string]$Exe,[string[]]$Arguments,[string]$Cwd,[int]$Seconds=600){
  $row=InvokeChild $Name $Exe $Arguments $Cwd $Seconds
  if($row.exitCode -ne 0 -or $row.timedOut -or -not $row.childExited){throw "child failed: $Name"}
}
function Download([string]$Name,$Asset){
  $target=Join-Path $stage $Name
  if(Test-Path $target){throw 'download target already exists'}
  Must "download-$Name" "$env:WINDIR/System32/curl.exe" @('-fsSL','--max-time','900','--output',$target,$Asset.url) $stage 910
  if((Hash $target) -cne $Asset.sha256){throw "download digest differs: $Name"}
  if($Asset.bytes -and (Get-Item $target).Length -ne $Asset.bytes){throw "download length differs: $Name"}
  return $target
}
function Inventory([string]$Root,[string]$Name){
  Must "inventory-$Name" $python @($verify,'manifest',$Root,"$out/$Name.json") $H
  return ,@(Get-Content "$out/$Name.json" -Raw | ConvertFrom-Json)
}
function CopyRuntime([string]$Runtime,[string]$Destination){
  if(Test-Path $Destination){throw 'cache fixture must be fresh'}
  New-Item -ItemType Directory $Destination | Out-Null
  Get-ChildItem -LiteralPath $Runtime -Force | Copy-Item -Destination $Destination -Recurse
}
$identity=@{executorSha=$env:GITHUB_SHA;coreSha=$env:INPUT_CORE_SHA;ownerRunId=$env:INPUT_OWNER_RUN_ID;ownerArtifactId=$env:INPUT_OWNER_ARTIFACT_ID;ownerArtifactSha256=$env:INPUT_OWNER_ARTIFACT_SHA256;ownerWorkflowHeadSha=$env:INPUT_OWNER_WORKFLOW_HEAD_SHA;ownerNativeCommit=$env:INPUT_OWNER_NATIVE_COMMIT;runId=$env:GITHUB_RUN_ID;runAttempt=$env:GITHUB_RUN_ATTEMPT}
SaveJson "$out/expected-identity.json" $identity
try {
  if(-not $IsWindows -or $env:GITHUB_REPOSITORY -cne 'leehack/llamadart-native' -or $env:GITHUB_EVENT_NAME -cne 'workflow_dispatch'){throw 'owner Windows manual workflow required'}
  if($env:GITHUB_REF -cne 'refs/heads/main' -or $env:GITHUB_SHA -cne $env:INPUT_EXPECTED_EXECUTOR_SHA){throw 'executor moved or is not owner main'}
  if($identity.coreSha -cne $spec.core_sha){throw 'core source differs from reviewed campaign'}
  if((& git -C $CoreCheckout rev-parse HEAD) -cne $spec.core_sha -or (& git -C $CoreCheckout rev-parse 'HEAD^{tree}') -cne $spec.core_tree){throw 'core checkout/tree identity differs'}
  # Validate own exact publication artifact access before SDK/model setup. prepare() is read-only and never publishes.
  $intake=Join-Path $stage 'intake'
  Must 'candidate-intake' $python @($verify,'intake','--core-sha',$identity.coreSha,'--owner-run-id',$identity.ownerRunId,'--owner-artifact-id',$identity.ownerArtifactId,'--owner-artifact-sha256',$identity.ownerArtifactSha256,'--owner-workflow-head-sha',$identity.ownerWorkflowHeadSha,'--owner-native-commit',$identity.ownerNativeCommit,'--directory',$intake) $H 1800
  Copy-Item "$intake/intake.json" "$out/intake.json"
  $candidateRuntime=Join-Path $intake 'windows-runtime'
  $candidateFiles=Inventory $candidateRuntime 'candidate-runtime-files'
  $sdkZip=Download 'sdk' $spec.assets.sdk
  $sdkRoot=Join-Path $stage 'sdk'
  Must 'extract-sdk' $python @($verify,'extract',$sdkZip,$sdkRoot,'--prefix','flutter/bin/cache/dart-sdk/') $H
  $dart=Join-Path $sdkRoot 'flutter/bin/cache/dart-sdk/bin/dart.exe'
  Must 'sdk-version' $dart @('--version') $stage
  if((Get-Content "$out/sdk-version.stdout.log" -Raw)+(Get-Content "$out/sdk-version.stderr.log" -Raw) -notmatch 'Dart SDK version: 3\.13\.1 '){throw 'pinned Dart version differs'}
  $oldPackage=Download 'old-package' $spec.assets.old_package
  $oldCore=Join-Path $stage 'old-core'
  Must 'extract-old-source' $python @($verify,'extract',$oldPackage,$oldCore) $H
  $oldSource=Inventory $oldCore 'old-source-before'
  if($oldSource.Count -ne $spec.old_source_files){throw 'published source inventory count differs'}
  foreach($proof in $spec.old_source_proofs.PSObject.Properties){if((Hash (Join-Path $oldCore $proof.Name)) -cne $proof.Value){throw 'published source proof differs'}}
  if((Get-Content "$oldCore/lib/src/hook/native_release_pins.dart" -Raw) -notmatch "const llamaCppTag = 'v0.8.0-1';"){throw 'published source is not the original old pin'}
  $oldUnsupportedCore=Join-Path $stage 'old-unsupported-core'
  Copy-Item -LiteralPath $oldCore -Destination $oldUnsupportedCore -Recurse
  $oldNative=Download 'old-runtime' $spec.assets.old_runtime
  $oldRuntime=Join-Path $stage 'old-runtime'
  Must 'extract-old-runtime' $python @($verify,'extract',$oldNative,$oldRuntime) $H
  $oldFiles=Inventory $oldRuntime 'old-runtime-files'
  $model=Download 'model' $spec.assets.model
  $newCore=Join-Path $stage 'new-core'
  $coreArchive=Join-Path $stage 'core-source.tar'
  Must 'export-core-source' (Get-Command git).Source @('-C',$CoreCheckout,'archive','--format=tar',"--output=$coreArchive",$spec.core_sha) $H
  Must 'extract-core-source' $python @($verify,'extract',$coreArchive,$newCore) $H
  $newSource=Inventory $newCore 'new-source-before'
  $aotHelper=Join-Path $CoreCheckout $spec.aot_helper
  . $aotHelper
  $applications=@{}
  foreach($mode in @('old-default','new-candidate','old-incompatible-candidate')){
    $app=Join-Path $stage $mode;New-Item -ItemType Directory $app | Out-Null
    $source=if($mode -eq 'new-candidate'){$newCore}elseif($mode -eq 'old-incompatible-candidate'){$oldUnsupportedCore}else{$oldCore}
    $pubspec="name: windows_consumer`nversion: 0.0.0`nenvironment:`n  sdk: '>=3.13.1 <4.0.0'`ndependencies:`n  llamadart:`n    path: '$($source.Replace('\','/'))'`n  crypto: $($spec.dependencies.crypto)`n  ffi: $($spec.dependencies.ffi)`n"
    $overlay=$mode -ne 'old-default'
    $pubspec+="hooks:`n  user_defines:`n    llamadart:`n"
    if($overlay){$pubspec+="      llamadart_native_path: '$($candidateRuntime.Replace('\','/'))'`n"}
    $pubspec+="      llamadart_native_runtimes: [llama_cpp]`n      llamadart_native_backends:`n        platforms:`n          windows-x64: [cpu]`n"
    [IO.File]::WriteAllText("$app/pubspec.yaml",$pubspec,[Text.UTF8Encoding]::new($false))
    Copy-Item "$app/pubspec.yaml" "$out/$mode-pubspec.yaml"
    Copy-Item "$H/payload/consumer.dart" "$app/load_stock.dart"
    Copy-Item (Join-Path $CoreCheckout $spec.observer) "$app/passive_modules.dart"
    SaveJson "$out/$mode-input-source.json" @{core=$source;coreSha=$spec.core_sha;oldPackageSha256=$spec.assets.old_package.sha256;consumerSource=$(if($mode -eq 'new-candidate'){'exact-core-git'}else{'published-0.17.5'});solePubspecOverlay=$overlay;pubspecSha256=Hash "$app/pubspec.yaml";payloadSha256=Hash "$app/load_stock.dart";observerSha256=Hash "$app/passive_modules.dart";aotHelperSha256=Hash $aotHelper;candidateDiagnostic=$overlay}
    # Keep this application's dependency resolution isolated and retain the exact resolved lock/config.
    $env:PUB_CACHE=Join-Path $stage "build-pub-$mode";New-Item -ItemType Directory $env:PUB_CACHE | Out-Null
    # InvokeChild owns a fresh cache per child; pub get and build each resolve independently, with both lock receipts retained.
    Must "$mode-pub" $dart @('pub','get') $app
    Copy-Item "$app/pubspec.lock" "$out/$mode-resolution-before.lock"
    Copy-Item "$app/.dart_tool/package_config.json" "$out/$mode-package-config.json"
    Must "$mode-build" $dart @('build','cli','-o','build/stock','-t','load_stock.dart') $app 1200
    Copy-Item "$app/pubspec.lock" "$out/$mode-pubspec.lock"
    Copy-Item "$app/.dart_tool/package_config.json" "$out/$mode-package-config.json" -Force
    $applications[$mode]=$app
  }
  $oldCacheArchive=Join-Path $oldCore '.dart_tool/llamadart/native_bundles/v0.8.0-1/windows-x64/llamadart-native-windows-x64-v0.8.0-1.tar.gz'
  if((Hash $oldCacheArchive) -cne $spec.assets.old_runtime.sha256){throw 'ordinary old hook did not acquire exact old archive'}
  SaveJson "$out/ordinary-old-delivery.json" @{tag='v0.8.0-1';archiveSha256=Hash $oldCacheArchive;expectedExactUrl=$spec.assets.old_runtime.url;sourcePinVerified=$true}
  $cache=$stage;$missing=Join-Path $stage 'missing-legacy-cache'
  $legacy=Join-Path $cache '.dart_tool/llamadart/native_bundles/v0.8.0-1/windows-x64/extracted'
  $v2=Join-Path $cache '.dart_tool/llamadart/native_bundles_v2/v0.8.0-3/windows-x64/extracted'
  $missingV2=Join-Path $missing '.dart_tool/llamadart/native_bundles_v2/v0.8.0-3/windows-x64/extracted'
  CopyRuntime $oldRuntime $legacy;CopyRuntime $candidateRuntime $v2;CopyRuntime $candidateRuntime $missingV2
  $sourceRoots=@($CoreCheckout,$oldCore,$oldUnsupportedCore,$newCore)+@($applications.Values)
  $cleanPath="$env:WINDIR/System32;$env:WINDIR";$buildPath=$env:Path
  foreach($name in $spec.cases){
    $aot=$name.EndsWith('-aot');$old=$name -in @('old-default','old-aot')
    $mode=if($name -eq 'old-aot'){'old-default'}elseif($name -eq 'new-aot'){'new-candidate'}else{$name}
    $app=$applications[$mode];$family=if($old){'old'}else{'candidate'}
    $arguments=@($mode,$model,$cache,$missing,$family)
    $expectedCache=if($old){$legacy}else{$v2}
    $expectedMissing=if($old){Join-Path $missing 'sdk'}else{$missingV2}
    $plan=$null
    Remove-Item Env:LLAMADART_NATIVE_LIB_DIR,Env:LLAMADART_BACKEND_MODULE_DIR -ErrorAction SilentlyContinue
    if($aot){
      $distribution=Join-Path $Scratch "distributed-$name";$cwd=Join-Path $Scratch "empty-$name"
      $plan=New-AotDistribution "$app/build/stock" $distribution $cwd $sourceRoots
      $env:Path=$cleanPath
      $row=InvokeChild $name "$distribution/bin/load_stock.exe" $arguments $cwd 900
      $row.cleanPath=$env:Path -ceq $cleanPath;$row.overridesAbsent=-not $env:LLAMADART_NATIVE_LIB_DIR -and -not $env:LLAMADART_BACKEND_MODULE_DIR
      $row.cwdEmpty=@(Get-ChildItem -LiteralPath $cwd -Force).Count -eq 0
      $row.distributionRoot=$distribution;$row.workingDirectory=$cwd;$row.sourceRoots=$sourceRoots
      $row.builtInventory=$plan.built.files;$row.distributionBefore=$plan.distributed.files;$row.distributionAfter=(Get-AotBundleManifest $distribution).files
      Assert-AotManifestEqual $plan.built (Get-AotBundleManifest "$app/build/stock")
      $env:Path=$buildPath
    }else{
      $row=InvokeChild $name $dart (@('run','load_stock.dart')+$arguments) $app 900
    }
    $row.cacheRoot=$cache;$row.expectedCache=$expectedCache;$row.expectedMissingCache=$expectedMissing
    $cases+=,$row
    if($row.timedOut -or -not $row.childExited){throw "consumer child did not finish: $name"}
    if($name -eq 'old-incompatible-candidate'){if($row.exitCode -eq 0){throw 'unsupported old/candidate pairing unexpectedly succeeded'}}
    elseif($row.exitCode -ne 0){throw "supported consumer child failed: $name"}
  }
  foreach($pair in @(@{root=$oldCore;before=$oldSource},@{root=$newCore;before=$newSource},@{root=$oldUnsupportedCore;before=$oldSource})){
    foreach($file in $pair.before){if((Hash (Join-Path $pair.root $file.relativePath)) -cne $file.sha256){throw 'consumer core source changed'}}
  }
  $completed=$true
} catch {$failure=$_.Exception.Message}
finally {
  $runtimeInventories=@{old=@($oldFiles | Where-Object {$_.relativePath -match '\.dll$'});candidate=@($candidateFiles | Where-Object {$_.relativePath -match '\.dll$'})}
  $terminal=@{identity=$identity;completed=$completed;failure=$failure;allChildrenExited=$childrenExited;cases=$cases;runtimeInventories=$runtimeInventories;sourceUnchanged=$completed;diagnosticCandidateOverride=$true;modelSha256=$(if($model -and (Test-Path $model)){Hash $model}else{$null});oldPackageSha256=$(if($oldPackage -and (Test-Path $oldPackage)){Hash $oldPackage}else{$null});sdkSha256=$(if($sdkZip -and (Test-Path $sdkZip)){Hash $sdkZip}else{$null});finalOrdinaryRepinQualified=$false;gpuQualified=$false;liveMtpQualified=$false}
  SaveJson "$out/terminal.json" $terminal
}
if(-not $completed){throw "qualification failed: $failure"}
# The workflow seals the evidence, then runs the independent verifier before upload.
