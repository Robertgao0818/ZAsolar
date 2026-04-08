param(
    [string]$ProcessName = "downloader"
)

$BM_CLICK = 0x00F5

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class GeidStart {
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
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);
}
"@

function Get-ClassName {
    param([IntPtr]$Handle)
    $sb = New-Object System.Text.StringBuilder 256
    [void][GeidStart]::GetClassName($Handle, $sb, $sb.Capacity)
    $sb.ToString()
}

function Get-WindowText {
    param([IntPtr]$Handle)
    $sb = New-Object System.Text.StringBuilder 256
    [void][GeidStart]::GetWindowText($Handle, $sb, $sb.Capacity)
    $sb.ToString()
}

$proc = Get-Process -Name $ProcessName -ErrorAction Stop | Select-Object -First 1
$procId = [uint32]$proc.Id
$form = [IntPtr]::Zero

$topCallback = [GeidStart+EnumWindowsProc]{
    param($hWnd, $lParam)
    $windowPid = [uint32]0
    [void][GeidStart]::GetWindowThreadProcessId($hWnd, [ref]$windowPid)
    if ($windowPid -eq $procId -and [GeidStart]::IsWindowVisible($hWnd)) {
        if ((Get-ClassName $hWnd) -eq "TForm1") {
            $script:form = $hWnd
            return $false
        }
    }
    return $true
}
[void][GeidStart]::EnumWindows($topCallback, [IntPtr]::Zero)

if ($form -eq [IntPtr]::Zero) {
    throw "Visible GEID form not found."
}

$button = [IntPtr]::Zero
$childCallback = [GeidStart+EnumWindowsProc]{
    param($hWnd, $lParam)
    if ((Get-ClassName $hWnd) -ne "TButton") {
        return $true
    }
    if ((Get-WindowText $hWnd) -eq "Start") {
        $script:button = $hWnd
        return $false
    }
    return $true
}
[void][GeidStart]::EnumChildWindows($form, $childCallback, [IntPtr]::Zero)

if ($button -eq [IntPtr]::Zero) {
    throw "Start button not found."
}

[void][GeidStart]::SendMessage($button, $BM_CLICK, [IntPtr]::Zero, [IntPtr]::Zero)
Start-Sleep -Milliseconds 500
Write-Output ("button_text={0}" -f (Get-WindowText $button))
