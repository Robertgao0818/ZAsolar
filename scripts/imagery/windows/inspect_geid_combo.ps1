param(
    [string]$ProcessName = "downloader"
)

$CB_GETCOUNT = 0x146
$CB_GETLBTEXT = 0x148

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class GeidCombo {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool EnumChildWindows(IntPtr hWnd, EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetClassName(IntPtr hWnd, StringBuilder lpClassName, int nMaxCount);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, StringBuilder lParam);
}
"@

function Get-ClassName {
    param([IntPtr]$Handle)
    $sb = New-Object System.Text.StringBuilder 256
    [void][GeidCombo]::GetClassName($Handle, $sb, $sb.Capacity)
    $sb.ToString()
}

$proc = Get-Process -Name $ProcessName -ErrorAction Stop | Select-Object -First 1
$procId = [uint32]$proc.Id
$form = [IntPtr]::Zero

$topCallback = [GeidCombo+EnumWindowsProc]{
    param($hWnd, $lParam)
    $windowPid = [uint32]0
    [void][GeidCombo]::GetWindowThreadProcessId($hWnd, [ref]$windowPid)
    if ($windowPid -eq $procId -and [GeidCombo]::IsWindowVisible($hWnd)) {
        if ((Get-ClassName $hWnd) -eq "TForm1") {
            $script:form = $hWnd
            return $false
        }
    }
    return $true
}
[void][GeidCombo]::EnumWindows($topCallback, [IntPtr]::Zero)

if ($form -eq [IntPtr]::Zero) {
    throw "Visible TForm1 not found."
}

$combo = [IntPtr]::Zero
$childCallback = [GeidCombo+EnumWindowsProc]{
    param($hWnd, $lParam)
    if ((Get-ClassName $hWnd) -eq "TComboBox") {
        $script:combo = $hWnd
        return $false
    }
    return $true
}
[void][GeidCombo]::EnumChildWindows($form, $childCallback, [IntPtr]::Zero)

if ($combo -eq [IntPtr]::Zero) {
    throw "TComboBox not found."
}

$count = [int][GeidCombo]::SendMessage($combo, $CB_GETCOUNT, [IntPtr]::Zero, (New-Object System.Text.StringBuilder 1))
Write-Output ("COUNT={0}" -f $count)
for ($i = 0; $i -lt $count; $i++) {
    $sb = New-Object System.Text.StringBuilder 512
    [void][GeidCombo]::SendMessage($combo, $CB_GETLBTEXT, [IntPtr]$i, $sb)
    Write-Output ("{0}:{1}" -f $i, $sb.ToString())
}
