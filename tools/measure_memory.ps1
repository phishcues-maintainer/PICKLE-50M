param(
    [Parameter(Mandatory = $true)][string]$Runtime,
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Threads = 4,
    [int]$NewTokens = 64,
    [int]$Iterations = 3
)

$runtimePath = (Resolve-Path -LiteralPath $Runtime).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$outPath = [IO.Path]::GetFullPath($Out)
$outDirectory = [IO.Path]::GetDirectoryName($outPath)
[IO.Directory]::CreateDirectory($outDirectory) | Out-Null
$stdoutPath = [IO.Path]::Combine($outDirectory, "memory-benchmark-stdout.json")
$stderrPath = [IO.Path]::Combine($outDirectory, "memory-benchmark-stderr.txt")
$tokens = "4068,793,3064,728,1178,98,1334,885,2079"
$arguments = @(
    "model-bench", "--model", $modelPath, "--tokens", $tokens,
    "--new-tokens", $NewTokens, "--iterations", $Iterations,
    "--threads", $Threads, "--kernel", "avx2"
)

$process = Start-Process -FilePath $runtimePath -ArgumentList $arguments -PassThru `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$peak = 0L
$samples = 0
while (-not $process.HasExited) {
    try {
        $process.Refresh()
        $peak = [Math]::Max($peak, $process.WorkingSet64)
        $samples += 1
    } catch {
        break
    }
    Start-Sleep -Milliseconds 10
}
$process.WaitForExit()
$process.Refresh()
$peak = [Math]::Max($peak, $process.PeakWorkingSet64)
if ($null -ne $process.ExitCode -and $process.ExitCode -ne 0) {
    throw "benchmark failed with exit code $($process.ExitCode): $([IO.File]::ReadAllText($stderrPath))"
}

$benchmark = [IO.File]::ReadAllText($stdoutPath) | ConvertFrom-Json
$decodeRate = if ($null -ne $benchmark.tokens_per_second) {
    $benchmark.tokens_per_second
} else {
    $benchmark.decode_tokens_per_second
}
$result = [ordered]@{
    format = "pickle-native-memory-sample-v2"
    platform = [Environment]::OSVersion.VersionString
    runtime = $runtimePath
    runtime_bytes = (Get-Item -LiteralPath $runtimePath).Length
    model = $modelPath
    model_bytes = (Get-Item -LiteralPath $modelPath).Length
    kernel = "avx2"
    worker_threads = $Threads
    prompt_tokens = $benchmark.prompt_tokens
    decode_tokens_per_iteration = $NewTokens
    iterations = $Iterations
    poll_interval_ms = 10
    samples = $samples
    peak_working_set_bytes = $peak
    peak_working_set_mib = $peak / 1MB
    decode_tokens_per_second = $decodeRate
    method = "PowerShell sampled Process.WorkingSet64 and read PeakWorkingSet64"
}
[IO.File]::WriteAllText($outPath, (($result | ConvertTo-Json -Depth 5) + "`n"))
$result | ConvertTo-Json -Depth 5
