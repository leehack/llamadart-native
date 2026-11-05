$ErrorActionPreference='Stop'
$PSNativeCommandUseErrorActionPreference=$false
$python=(Get-Command python).Source
$verify=Join-Path $PSScriptRoot 'verify_evidence.py'
$tokens=$null;$errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile(
  (Join-Path $PSScriptRoot 'qualification.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'qualification PowerShell syntax errors'}
# Import complete function definitions without running owner intake or consumers.
foreach($name in @('Hash','Download')){
  $definitions=@($ast.FindAll({param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq $name
  },$false))
  if($definitions.Count -ne 1){throw "expected one production function: $name"}
  . ([scriptblock]::Create($definitions[0].Extent.Text))
}
function Must([string]$Name,[string]$Exe,[string[]]$Arguments,[string]$Cwd,[int]$Seconds=600){
  if($Exe -notlike '*/System32/curl.exe'){throw 'unexpected fixture child'}
  $script:downloadCalls++
  $outputIndex=[array]::IndexOf($Arguments,'--output')
  if($outputIndex -lt 0){throw 'download output missing'}
  Copy-Item -LiteralPath $Arguments[-1] -Destination $Arguments[$outputIndex+1]
}
function ExpectFailure([scriptblock]$Action,[string]$Pattern){
  $message=$null
  try { & $Action | Out-Null } catch { $message=$_.Exception.Message }
  if(-not $message -or $message -notmatch $Pattern){throw "expected refusal: $Pattern; got: $message"}
}
$scratch=Join-Path ([IO.Path]::GetTempPath()) ('consumer-staging-test-'+[guid]::NewGuid())
$stage=Join-Path $scratch 'stage';$downloads=Join-Path $stage 'downloads'
$fixtures=Join-Path $scratch 'fixtures'
$script:downloadCalls=0
try {
  New-Item -ItemType Directory $stage,$downloads,$fixtures | Out-Null
  @'
import io, sys, tarfile, zipfile
from pathlib import Path
root = Path(sys.argv[1])
with zipfile.ZipFile(root / 'sdk.zip', 'w') as archive:
    archive.writestr('flutter/bin/cache/dart-sdk/bin/dart.exe', b'synthetic SDK text')
    archive.writestr('flutter/unselected.txt', b'not selected')
with tarfile.open(root / 'runtime.tar.gz', 'w:gz') as archive:
    member = tarfile.TarInfo('./llamadart.dll')
    data = b'synthetic runtime text'
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))
'@ | & $python - $fixtures
  if($LASTEXITCODE -ne 0){throw 'synthetic archive creation failed'}
  foreach($case in @(
    @{name='sdk';fixture='sdk.zip';member='flutter/bin/cache/dart-sdk/bin/dart.exe';text='synthetic SDK text';extra=@('--prefix','flutter/bin/cache/dart-sdk/')},
    @{name='old-runtime';fixture='runtime.tar.gz';member='llamadart.dll';text='synthetic runtime text';extra=@()}
  )){
    $fixture=Join-Path $fixtures $case.fixture
    $asset=@{url=$fixture;sha256=Hash $fixture;bytes=(Get-Item $fixture).Length}
    $archive=Download $case.name $asset
    $destination=Join-Path $stage $case.name
    if($archive -cne (Join-Path $downloads $case.name)){throw 'download escaped downloads directory'}
    & $python $verify extract $archive $destination @($case.extra)
    if($LASTEXITCODE -ne 0){throw 'synthetic extraction failed'}
    if([IO.File]::ReadAllText((Join-Path $destination $case.member)) -cne $case.text){throw 'extracted content differs'}
    if((Hash $archive) -cne $asset.sha256){throw 'extraction changed archive'}
    if($case.name -eq 'sdk' -and (Test-Path (Join-Path $destination 'flutter/unselected.txt'))){throw 'SDK prefix ignored'}
    $calls=$script:downloadCalls
    ExpectFailure { Download $case.name $asset } 'download target already exists'
    if($script:downloadCalls -ne $calls -or (Hash $archive) -cne $asset.sha256){throw 'existing download overwritten'}
    $refusal=& $python $verify extract $archive $destination @($case.extra) 2>&1
    if($LASTEXITCODE -eq 0 -or "$refusal" -notmatch 'fresh extraction directory required'){throw 'existing extraction accepted'}
    # Reproduce the old archive/output collision; the extractor must still refuse it.
    $refusal=& $python $verify extract $archive $archive 2>&1
    if($LASTEXITCODE -eq 0 -or "$refusal" -notmatch 'fresh extraction directory required'){throw 'archive/output collision accepted'}
    if((Hash $archive) -cne $asset.sha256){throw 'refusal changed archive'}
  }
  $asset=@{url=(Join-Path $fixtures 'sdk.zip');sha256=('0'*64);bytes=1}
  ExpectFailure { Download 'bad-hash' $asset } 'download digest differs'
  $asset.sha256=Hash $asset.url
  ExpectFailure { Download 'bad-length' $asset } 'download length differs'
  Write-Output 'Synthetic production download/extraction staging checks passed'
} finally {
  if(Test-Path -LiteralPath $scratch){Remove-Item -LiteralPath $scratch -Recurse -Force}
}
