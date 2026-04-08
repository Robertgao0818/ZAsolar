# extend_cipher_key.ps1 — run ONCE on Windows to extend the GEID XOR key.
#
# Why: 97 / 83579 Joburg-batch1 tiles failed because some z=21 JPEGs are
# 20-23 KB and exceed our 19759-byte cipher key. The key is fixed (not
# session-specific), so extending it once permanently solves it for all
# future batches.
#
# How: GEID's CLI mode (https://www.allmapsoft.com/geid/commandline.htm)
# downloads the target tile and writes the DECRYPTED JPG to disk. We
# already have the ENCRYPTED wire body (saved on Linux into
# data/geid_protocol/key_extension/wire_*.bin). XOR-ing the two recovers
# the key bytes for positions 19759..23241 — that's all we need.
#
# Usage (run from anywhere on Windows; assumes repo cloned to a path
# accessible from Windows):
#
#     powershell -ExecutionPolicy Bypass `
#       -File \\wsl$\Ubuntu\home\gaosh\projects\ZAsolar\geid_reverse_engineering\windows\extend_cipher_key.ps1
#
# Or — even simpler — invoke downloader.exe directly from WSL bash:
#     "/mnt/c/allmapsoft/geid/downloader.exe" task 21 21 L R T B "C:\Temp\geid_key_ext"
# (CLI mode runs headless and auto-exits; verified on GEID 6.48 / 2026-04-08)
#
# Or copy this file to C:\Temp first if UNC paths give trouble.
#
# Adjust GEID_EXE below if your install path differs.

$ErrorActionPreference = 'Stop'

# === Config ===
$GEID_EXE = 'C:\Program Files (x86)\AllmapsoftInc\Google Earth Images Downloader 6.48\downloader.exe'
if (-not (Test-Path $GEID_EXE)) {
  # Try alternative install paths
  $alts = @(
    'C:\Program Files\AllmapsoftInc\Google Earth Images Downloader 6.48\downloader.exe',
    'C:\Program Files (x86)\Google Earth Images Downloader\downloader.exe',
    'C:\Program Files\Google Earth Images Downloader\downloader.exe'
  )
  foreach ($p in $alts) {
    if (Test-Path $p) { $GEID_EXE = $p; break }
  }
}
if (-not (Test-Path $GEID_EXE)) {
  throw "downloader.exe not found. Set `$GEID_EXE at the top of this script to your GEID install path."
}
Write-Output "GEID exe: $GEID_EXE"

# Target tile coordinates (must match what wire_*.bin was captured for)
# Source: data/geid_protocol/key_extension/target_meta.json
$TARGET_X = 605987
$TARGET_Y = 447923
$TARGET_Z = 21
# Bbox slightly expanded around the target tile so GEID definitely picks it up
# (may also download 1-3 neighbors, which is harmless)
$LON_L = 28.0490913
$LON_R = 28.0495033
$LAT_T = -26.2174644
$LAT_B = -26.2178764
$TASK = "key_ext_${TARGET_X}_${TARGET_Y}"

$SAVE_ROOT = 'C:\Temp\geid_key_ext'
New-Item -Path $SAVE_ROOT -ItemType Directory -Force | Out-Null

# Expected output file from GEID CLI
$EXPECTED_JPG = Join-Path $SAVE_ROOT "$TASK\$TARGET_Z\$TARGET_X\ges_${TARGET_X}_${TARGET_Y}_${TARGET_Z}.jpg"
Write-Output "expected jpg: $EXPECTED_JPG"

# Clean any prior attempt so we know whether GEID actually wrote a fresh file
if (Test-Path $EXPECTED_JPG) { Remove-Item $EXPECTED_JPG -Force }

# === Invoke GEID CLI ===
# downloader.exe para1 para2 para3 para4 para5 para6 para7 para8 [para9]
#                task   zfrom zto   L     R     T     B     savepath  date
$cliArgs = @(
  $TASK,
  "$TARGET_Z", "$TARGET_Z",
  "$LON_L", "$LON_R", "$LAT_T", "$LAT_B",
  $SAVE_ROOT
)
Write-Output ""
Write-Output "Running: $GEID_EXE $($cliArgs -join ' ')"
Write-Output ""

$proc = Start-Process -FilePath $GEID_EXE -ArgumentList $cliArgs -PassThru
Write-Output "downloader.exe pid=$($proc.Id), waiting up to 90s..."

# Wait up to 90s for either the file to appear or the process to exit
$deadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $deadline) {
  if (Test-Path $EXPECTED_JPG) {
    $sz = (Get-Item $EXPECTED_JPG).Length
    if ($sz -gt 0) {
      Write-Output "✓ tile written: $EXPECTED_JPG ($sz bytes)"
      break
    }
  }
  if ($proc.HasExited) {
    Write-Output "downloader.exe exited (code=$($proc.ExitCode))"
    break
  }
  Start-Sleep -Milliseconds 500
}

# Try to close downloader if still running
if (-not $proc.HasExited) {
  Write-Warning "downloader still running, attempting close..."
  Start-Sleep -Seconds 2
  if (-not $proc.HasExited) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
  }
}

if (-not (Test-Path $EXPECTED_JPG)) {
  Write-Error @"
GEID CLI did not produce the expected JPG.
Possible causes:
  1. CLI mode requires a different argument format in your GEID version
  2. License dialog may have popped (check Windows desktop)
  3. The bbox is too small — try widening LON_L/LON_R/LAT_T/LAT_B in this script

FALLBACK: download the tile manually in the GEID GUI:
  - Set bbox: lon $LON_L .. $LON_R, lat $LAT_B .. $LAT_T
  - Zoom: $TARGET_Z to $TARGET_Z
  - Save folder: $SAVE_ROOT\$TASK
  - Click Start, wait for "Task finished", click OK
Then re-run the key extender (Linux-side) pointing at the resulting JPG.
"@
  exit 1
}

# === Sanity check + copy out for the Linux-side extender ===
$jpgBytes = [System.IO.File]::ReadAllBytes($EXPECTED_JPG)
Write-Output "JPG size: $($jpgBytes.Length) bytes"
if ($jpgBytes[0] -ne 0xFF -or $jpgBytes[1] -ne 0xD8 -or $jpgBytes[2] -ne 0xFF) {
  Write-Error "✗ JPG does not start with FFD8FF — corrupt or wrong encoding"
  exit 2
}
Write-Output "✓ JPG starts with FFD8FF (valid SOI marker)"

# Copy into the WSL repo so the extender script finds it
$wslDest = '\\wsl$\Ubuntu\home\gaosh\projects\ZAsolar\geid_reverse_engineering\examples\key_extension'
if (Test-Path $wslDest) {
  $destFile = Join-Path $wslDest "plain_${TARGET_X}_${TARGET_Y}_${TARGET_Z}.jpg"
  Copy-Item -Path $EXPECTED_JPG -Destination $destFile -Force
  Write-Output "✓ copied to: $destFile"
} else {
  Write-Output "WSL path not reachable; copy manually:"
  Write-Output "  src: $EXPECTED_JPG"
  Write-Output "  dst: <repo>/geid_reverse_engineering/examples/key_extension/plain_${TARGET_X}_${TARGET_Y}_${TARGET_Z}.jpg"
}

Write-Output ""
Write-Output "DONE on Windows side. Next, on Linux/WSL run:"
Write-Output "  python3 geid_reverse_engineering/python/extend_cipher_key.py"
