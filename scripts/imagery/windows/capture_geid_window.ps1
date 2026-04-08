param(
    [string]$ProcessName = "downloader",
    [string]$OutputPath = "\\wsl.localhost\Ubuntu\tmp\geid_window.png",
    [string]$WindowHandle = ""
)

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class Win32Capture {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }
}
"@

$proc = Get-Process -Name $ProcessName -ErrorAction Stop | Select-Object -First 1
$hwnd = [IntPtr]::Zero
if ($WindowHandle) {
    $clean = $WindowHandle.Trim()
    if ($clean.StartsWith("0x")) {
        $hwnd = [IntPtr]([Convert]::ToInt64($clean.Substring(2), 16))
    } else {
        $hwnd = [IntPtr]([Convert]::ToInt64($clean, 10))
    }
} else {
    $procId = [uint32]$proc.Id
    $candidates = New-Object System.Collections.Generic.List[object]

    $callback = [Win32Capture+EnumWindowsProc]{
        param($hWnd, $lParam)
        $windowPid = [uint32]0
        [void][Win32Capture]::GetWindowThreadProcessId($hWnd, [ref]$windowPid)
        if ($windowPid -ne $procId -or -not [Win32Capture]::IsWindowVisible($hWnd)) {
            return $true
        }

        $rect = New-Object Win32Capture+RECT
        if (-not [Win32Capture]::GetWindowRect($hWnd, [ref]$rect)) {
            return $true
        }

        $width = $rect.Right - $rect.Left
        $height = $rect.Bottom - $rect.Top
        if ($width -le 50 -or $height -le 50) {
            return $true
        }

        $candidates.Add([PSCustomObject]@{
            Handle = $hWnd
            Width = $width
            Height = $height
            Area = $width * $height
        }) | Out-Null
        return $true
    }

    [void][Win32Capture]::EnumWindows($callback, [IntPtr]::Zero)

    if ($candidates.Count -eq 0) {
        throw "No visible top-level windows found for '$ProcessName'."
    }

    $hwnd = ($candidates | Sort-Object Area -Descending | Select-Object -First 1).Handle
}

$rect = New-Object Win32Capture+RECT
if (-not [Win32Capture]::GetWindowRect($hwnd, [ref]$rect)) {
    throw "Failed to read window rect."
}

$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top

$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)

$bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()

Write-Output $OutputPath
