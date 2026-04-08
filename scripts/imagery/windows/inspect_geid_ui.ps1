param(
    [string]$ProcessName = "downloader"
)

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class Win32Ui {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool EnumChildWindows(IntPtr hWnd, EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern int GetWindowTextLength(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetClassName(IntPtr hWnd, StringBuilder lpClassName, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }
}
"@

function Get-WindowText {
    param([IntPtr]$Handle)
    $len = [Win32Ui]::GetWindowTextLength($Handle)
    $sb = New-Object System.Text.StringBuilder ([Math]::Max($len + 1, 256))
    [void][Win32Ui]::GetWindowText($Handle, $sb, $sb.Capacity)
    $sb.ToString()
}

function Get-ClassName {
    param([IntPtr]$Handle)
    $sb = New-Object System.Text.StringBuilder 256
    [void][Win32Ui]::GetClassName($Handle, $sb, $sb.Capacity)
    $sb.ToString()
}

function Get-RectString {
    param([IntPtr]$Handle)
    $rect = New-Object Win32Ui+RECT
    if ([Win32Ui]::GetWindowRect($Handle, [ref]$rect)) {
        return "$($rect.Left),$($rect.Top),$($rect.Right),$($rect.Bottom)"
    }
    return ""
}

$proc = Get-Process -Name $ProcessName -ErrorAction Stop | Select-Object -First 1
$procId = [uint32]$proc.Id

$topWindows = New-Object System.Collections.Generic.List[IntPtr]
$topCallback = [Win32Ui+EnumWindowsProc]{
    param($hWnd, $lParam)
    $windowPid = [uint32]0
    [void][Win32Ui]::GetWindowThreadProcessId($hWnd, [ref]$windowPid)
    if ($windowPid -eq $procId) {
        $topWindows.Add($hWnd) | Out-Null
    }
    return $true
}
[void][Win32Ui]::EnumWindows($topCallback, [IntPtr]::Zero)

$rows = New-Object System.Collections.Generic.List[object]

foreach ($top in $topWindows) {
    $rows.Add([PSCustomObject]@{
        level   = 0
        handle  = ('0x{0:X}' -f $top.ToInt64())
        class   = Get-ClassName $top
        text    = Get-WindowText $top
        visible = [Win32Ui]::IsWindowVisible($top)
        rect    = Get-RectString $top
    }) | Out-Null

    $childCallback = [Win32Ui+EnumWindowsProc]{
        param($hWnd, $lParam)
        $rows.Add([PSCustomObject]@{
            level   = 1
            handle  = ('0x{0:X}' -f $hWnd.ToInt64())
            class   = Get-ClassName $hWnd
            text    = Get-WindowText $hWnd
            visible = [Win32Ui]::IsWindowVisible($hWnd)
            rect    = Get-RectString $hWnd
        }) | Out-Null
        return $true
    }

    [void][Win32Ui]::EnumChildWindows($top, $childCallback, [IntPtr]::Zero)
}

$rows | Format-Table -AutoSize
