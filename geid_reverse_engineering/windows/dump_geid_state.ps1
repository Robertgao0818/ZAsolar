param([int]$ProcessId)

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class W {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
    [DllImport("user32.dll")]
    public static extern bool EnumChildWindows(IntPtr hWnd, EnumWindowsProc lpEnumFunc, IntPtr lParam);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetClassName(IntPtr hWnd, StringBuilder lpClassName, int nMaxCount);
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);
}
"@

function GetText($h) {
    $sb = New-Object System.Text.StringBuilder 1024
    [void][W]::GetWindowText($h, $sb, 1024)
    return $sb.ToString()
}
function GetClass($h) {
    $sb = New-Object System.Text.StringBuilder 256
    [void][W]::GetClassName($h, $sb, 256)
    return $sb.ToString()
}

$forms = New-Object System.Collections.ArrayList
$cb1 = [W+EnumWindowsProc]{
    param($h, $l)
    $pid_out = [uint32]0
    [void][W]::GetWindowThreadProcessId($h, [ref]$pid_out)
    if ($pid_out -eq $ProcessId -and [W]::IsWindowVisible($h)) {
        [void]$forms.Add(@{Handle=$h; Class=(GetClass $h); Text=(GetText $h)})
    }
    return $true
}
[void][W]::EnumWindows($cb1, [IntPtr]::Zero)
Write-Host "=== Top-level visible windows for PID $ProcessId ==="
foreach ($f in $forms) { Write-Host ("  {0} class={1} text='{2}'" -f $f.Handle, $f.Class, $f.Text) }

# For each form, dump children
foreach ($f in $forms) {
    if ($f.Class -ne "TForm1") { continue }
    Write-Host "`n=== Children of TForm1 ($($f.Handle)) ==="
    $children = New-Object System.Collections.ArrayList
    $cb2 = [W+EnumWindowsProc]{
        param($h, $l)
        [void]$children.Add(@{Handle=$h; Class=(GetClass $h); Text=(GetText $h)})
        return $true
    }
    [void][W]::EnumChildWindows($f.Handle, $cb2, [IntPtr]::Zero)
    foreach ($c in $children) {
        Write-Host ("  {0} class={1} text='{2}'" -f $c.Handle, $c.Class, $c.Text)
    }
}
