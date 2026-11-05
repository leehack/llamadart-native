param([switch]$InjectUnexpectedFailure)
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
  # Execute the production selector and helper call, leaving only the external
  # core helper boundary synthetic. SDK build roots contain a bundle subdirectory.
  $selectors=@($ast.FindAll({param($node)
    $node -is [Management.Automation.Language.AssignmentStatementAst] -and
      $node.Left.Extent.Text -ceq '$builtBundle'
  },$true))
  $plans=@($ast.FindAll({param($node)
    $node -is [Management.Automation.Language.AssignmentStatementAst] -and
      $node.Left.Extent.Text -ceq '$plan' -and
      $node.Right.Extent.Text.StartsWith('New-AotDistribution ')
  },$true))
  $copies=@($ast.FindAll({param($node)
    $node -is [Management.Automation.Language.CommandAst] -and
      $node.GetCommandName() -ceq 'New-AotDistribution'
  },$true))
  $rechecks=@($ast.FindAll({param($node)
    $node -is [Management.Automation.Language.CommandAst] -and
      $node.GetCommandName() -ceq 'Get-AotBundleManifest' -and
      $node.CommandElements[1].Extent.Text -ceq '$builtBundle'
  },$true))
  if($selectors.Count -ne 1 -or $plans.Count -ne 1 -or $copies.Count -ne 1 -or $rechecks.Count -ne 1){throw 'production AOT bundle-root callsites missing or ambiguous'}
  if($copies[0].CommandElements[1].Extent.Text -cne '$builtBundle'){throw 'AOT copy bypassed the bound bundle root'}
  $app=Join-Path $stage 'aot-app';$bundle=Join-Path $app 'build/stock/bundle'
  New-Item -ItemType Directory (Join-Path $bundle 'bin'),(Join-Path $bundle 'lib/nested') | Out-Null
  foreach($name in @('bin/load_stock.exe','lib/nested/sidecar.json','.hidden-sidecar')){
    [IO.File]::WriteAllText((Join-Path $bundle $name),"synthetic $name")
  }
  [IO.File]::SetAttributes((Join-Path $bundle '.hidden-sidecar'),[IO.FileAttributes]::Hidden)
  [IO.File]::WriteAllText((Join-Path $app 'build/stock/build-only.txt'),'not a shipping file')
  $distribution=Join-Path $scratch 'distributed-aot';$cwd=Join-Path $scratch 'empty-aot'
  $sourceRoots=@($app);$expectedDistribution=$distribution;$script:aotCalls=0
  function New-AotDistribution([string]$BuiltBundle,[string]$Distribution,[string]$WorkingDirectory,[string[]]$SourceRoots){
    $script:aotCalls++
    if(-not (Test-Path -LiteralPath (Join-Path $BuiltBundle 'bin/load_stock.exe') -PathType Leaf)){throw 'built AOT executable missing'}
    if($Distribution -cne $expectedDistribution -or $WorkingDirectory -cne $cwd -or $SourceRoots.Count -ne 1 -or $SourceRoots[0] -cne $app){throw 'AOT distribution arguments changed'}
    $files=@(Get-ChildItem -LiteralPath $BuiltBundle -File -Recurse -Force | ForEach-Object {
      $_.FullName.Substring($BuiltBundle.Length+1).Replace('\','/')
    } | Sort-Object)
    if(($files -join '|') -cne '.hidden-sidecar|bin/load_stock.exe|lib/nested/sidecar.json'){throw 'AOT selected an incomplete bundle or its build parent'}
    return [pscustomobject]@{built=[pscustomobject]@{files=$files}}
  }
  $selection=[scriptblock]::Create($selectors[0].Extent.Text)
  $copy=[scriptblock]::Create($plans[0].Extent.Text)
  . $selection
  . $copy
  if($script:aotCalls -ne 1 -or $plan.built.files.Count -ne 3){throw 'production AOT copy callsite did not reach the boundary'}
  # Reproduce the actual SDK-parent failure through the same production callsite.
  $parentSelection=[scriptblock]::Create($selectors[0].Extent.Text.Replace('build/stock/bundle','build/stock'))
  if($parentSelection.ToString() -ceq $selection.ToString()){throw 'AOT parent-root control did not mutate the selector'}
  ExpectFailure { . $parentSelection; . $copy } 'built AOT executable missing'
  if($script:aotCalls -ne 2){throw 'AOT parent-root control did not reach the boundary'}
  if($InjectUnexpectedFailure){throw 'injected unexpected staging assertion failure'}
  Write-Output 'Synthetic production download/extraction and AOT bundle selection checks passed'
} finally {
  if(Test-Path -LiteralPath $scratch){Remove-Item -LiteralPath $scratch -Recurse -Force}
}
# Expected extraction refusals leave a nonzero native status; all checks and cleanup passed.
exit 0
