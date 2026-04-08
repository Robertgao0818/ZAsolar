param(
    [Parameter(Mandatory = $true)]
    [string]$TasksCsv,
    [string]$ExePath = "C:\allmapsoft\geid\downloader.exe",
    [switch]$ForceStopCurrentTask,
    [switch]$StartTasks,
    [int]$PollSeconds = 5,
    [int]$StartTimeoutSeconds = 30,
    [int]$CompletionTimeoutSeconds = 7200
)

$WM_SETTEXT = 0x000C
$BM_CLICK = 0x00F5
$SW_RESTORE = 9
$MOUSEEVENTF_LEFTDOWN = 0x0002
$MOUSEEVENTF_LEFTUP = 0x0004

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class GeidWin32 {
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

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, string lParam);

    [DllImport("user32.dll")]
    public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int X, int Y);

    [DllImport("user32.dll")]
    public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);

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
    $len = [GeidWin32]::GetWindowTextLength($Handle)
    $sb = New-Object System.Text.StringBuilder ([Math]::Max($len + 1, 256))
    [void][GeidWin32]::GetWindowText($Handle, $sb, $sb.Capacity)
    $sb.ToString()
}

function Get-ClassName {
    param([IntPtr]$Handle)
    $sb = New-Object System.Text.StringBuilder 256
    [void][GeidWin32]::GetClassName($Handle, $sb, $sb.Capacity)
    $sb.ToString()
}

function Get-RectObject {
    param([IntPtr]$Handle)
    $rect = New-Object GeidWin32+RECT
    if (-not [GeidWin32]::GetWindowRect($Handle, [ref]$rect)) {
        throw "Failed to read rect for handle $Handle"
    }
    [PSCustomObject]@{
        Left = $rect.Left
        Top = $rect.Top
        Right = $rect.Right
        Bottom = $rect.Bottom
        Width = $rect.Right - $rect.Left
        Height = $rect.Bottom - $rect.Top
    }
}

function Get-GeidProcess {
    $proc = Get-Process -Name downloader -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $proc) {
        return $proc
    }

    if (-not (Test-Path $ExePath)) {
        throw "GEID executable not found: $ExePath"
    }

    Start-Process -FilePath $ExePath | Out-Null
    Start-Sleep -Seconds 2

    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        $proc = Get-Process -Name downloader -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $proc) {
            return $proc
        }
        Start-Sleep -Milliseconds 500
    }

    throw "Failed to launch downloader.exe"
}

function Get-GeidTopWindows {
    param([uint32]$ProcessId)

    $rows = New-Object System.Collections.Generic.List[object]
    $callback = [GeidWin32+EnumWindowsProc]{
        param($hWnd, $lParam)
        $windowPid = [uint32]0
        [void][GeidWin32]::GetWindowThreadProcessId($hWnd, [ref]$windowPid)
        if ($windowPid -ne $ProcessId) {
            return $true
        }

        $rect = Get-RectObject $hWnd
        $rows.Add([PSCustomObject]@{
            Handle = $hWnd
            Class = Get-ClassName $hWnd
            Text = Get-WindowText $hWnd
            Visible = [GeidWin32]::IsWindowVisible($hWnd)
            Left = $rect.Left
            Top = $rect.Top
            Width = $rect.Width
            Height = $rect.Height
            Area = $rect.Width * $rect.Height
        }) | Out-Null
        return $true
    }

    [void][GeidWin32]::EnumWindows($callback, [IntPtr]::Zero)
    return $rows
}

function Get-GeidFormHandle {
    param([uint32]$ProcessId)

    $windows = Get-GeidTopWindows -ProcessId $ProcessId
    $form = $windows |
        Where-Object { $_.Class -eq "TForm1" -and $_.Visible -and $_.Area -gt 0 } |
        Sort-Object Area -Descending |
        Select-Object -First 1

    if ($null -eq $form) {
        throw "Visible GEID form (TForm1) not found."
    }
    return [IntPtr]$form.Handle
}

function Focus-GeidForm {
    param([IntPtr]$FormHandle)
    [void][GeidWin32]::ShowWindow($FormHandle, $SW_RESTORE)
    [void][GeidWin32]::SetForegroundWindow($FormHandle)
    Start-Sleep -Milliseconds 300
}

function Get-GeidControls {
    param([IntPtr]$FormHandle)

    $formRect = Get-RectObject $FormHandle
    $controls = New-Object System.Collections.Generic.List[object]
    $callback = [GeidWin32+EnumWindowsProc]{
        param($hWnd, $lParam)
        $rect = Get-RectObject $hWnd
        $controls.Add([PSCustomObject]@{
            Handle = $hWnd
            Class = Get-ClassName $hWnd
            Text = Get-WindowText $hWnd
            Left = $rect.Left - $formRect.Left
            Top = $rect.Top - $formRect.Top
            Right = $rect.Right - $formRect.Left
            Bottom = $rect.Bottom - $formRect.Top
            Width = $rect.Width
            Height = $rect.Height
        }) | Out-Null
        return $true
    }
    [void][GeidWin32]::EnumChildWindows($FormHandle, $callback, [IntPtr]::Zero)
    return $controls
}

function Resolve-GeidControlMap {
    param([object[]]$Controls)

    $edits = $Controls | Where-Object { $_.Class -eq "TEdit" }
    $buttons = $Controls | Where-Object { $_.Class -eq "TButton" }

    $taskName = $edits | Where-Object { $_.Top -ge 50 -and $_.Top -le 100 } | Select-Object -First 1
    $dateEdit = $edits | Where-Object { $_.Top -ge 120 -and $_.Top -le 170 } | Select-Object -First 1
    $zoomEdits = $edits |
        Where-Object { $_.Top -ge 170 -and $_.Top -le 230 -and $_.Width -le 60 } |
        Sort-Object Left
    $coordEdits = $edits |
        Where-Object { $_.Top -ge 240 -and $_.Top -le 340 -and $_.Width -ge 100 } |
        Sort-Object Top, Left
    $saveTo = $edits | Where-Object { $_.Top -ge 330 -and $_.Top -le 410 } | Select-Object -First 1
    $startButton = $buttons | Where-Object { $_.Top -ge 400 } | Select-Object -First 1

    if ($zoomEdits.Count -ne 2) {
        throw "Expected 2 zoom edits, got $($zoomEdits.Count)"
    }
    if ($coordEdits.Count -ne 4) {
        throw "Expected 4 coordinate edits, got $($coordEdits.Count)"
    }
    if ($null -eq $taskName -or $null -eq $saveTo -or $null -eq $startButton) {
        throw "Failed to resolve one or more GEID controls."
    }

    return [ordered]@{
        TaskName = $taskName
        Date = $dateEdit
        ZoomFrom = $zoomEdits[0]
        ZoomTo = $zoomEdits[1]
        LeftLongitude = $coordEdits[0]
        RightLongitude = $coordEdits[1]
        TopLatitude = $coordEdits[2]
        BottomLatitude = $coordEdits[3]
        SaveTo = $saveTo
        StartButton = $startButton
    }
}

function Set-EditText {
    param(
        [IntPtr]$Handle,
        [string]$Value
    )
    [void][GeidWin32]::SendMessage($Handle, $WM_SETTEXT, [IntPtr]::Zero, $Value)
    Start-Sleep -Milliseconds 120
    if ((Get-WindowText $Handle) -eq $Value) {
        return
    }

    $rect = Get-RectObject $Handle
    $x = [int]($rect.Left + [Math]::Floor($rect.Width / 2))
    $y = [int]($rect.Top + [Math]::Floor($rect.Height / 2))
    [void][GeidWin32]::SetCursorPos($x, $y)
    Start-Sleep -Milliseconds 120
    [GeidWin32]::mouse_event($MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
    [GeidWin32]::mouse_event($MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 120

    Set-Clipboard -Value $Value
    $shell = New-Object -ComObject WScript.Shell
    $shell.SendKeys('^a')
    Start-Sleep -Milliseconds 80
    $shell.SendKeys('^v')
    Start-Sleep -Milliseconds 180
}

function Click-Button {
    param([IntPtr]$Handle)
    [void][GeidWin32]::SendMessage($Handle, $BM_CLICK, [IntPtr]::Zero, [IntPtr]::Zero)
    Start-Sleep -Milliseconds 250
}

function Ensure-Directory {
    param([string]$PathValue)
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return
    }
    New-Item -Path $PathValue -ItemType Directory -Force | Out-Null
}

function Dump-VisibleWindows {
    param([uint32]$ProcessId)
    Get-GeidTopWindows -ProcessId $ProcessId |
        Where-Object { $_.Visible -and $_.Area -gt 0 } |
        Sort-Object Top, Left |
        Select-Object Class, Text, Left, Top, Width, Height |
        Format-Table -AutoSize |
        Out-String
}

function Wait-ForDownload {
    param(
        [IntPtr]$StartButtonHandle,
        [uint32]$ProcessId,
        [int]$PollIntervalSeconds,
        [int]$StartTimeout,
        [int]$CompletionTimeout
    )

    $runningSeen = $false
    $startDeadline = (Get-Date).AddSeconds($StartTimeout)
    while ((Get-Date) -lt $startDeadline) {
        $text = Get-WindowText $StartButtonHandle
        if ($text -eq "Stop") {
            $runningSeen = $true
            break
        }
        Start-Sleep -Seconds 1
    }

    if (-not $runningSeen) {
        Write-Warning "Start button never changed to 'Stop'. Visible windows:"
        Write-Host (Dump-VisibleWindows -ProcessId $ProcessId)
        return
    }

    $completionDeadline = (Get-Date).AddSeconds($CompletionTimeout)
    while ((Get-Date) -lt $completionDeadline) {
        $text = Get-WindowText $StartButtonHandle
        if ($text -eq "Start") {
            return
        }
        Start-Sleep -Seconds $PollIntervalSeconds
    }

    throw "Timed out waiting for GEID task completion."
}

function Wait-ForButtonText {
    param(
        [IntPtr]$Handle,
        [string]$ExpectedText,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ((Get-WindowText $Handle) -eq $ExpectedText) {
            return
        }
        Start-Sleep -Milliseconds 500
    }

    throw "Timed out waiting for button text '$ExpectedText'."
}

$tasks = Import-Csv -Path $TasksCsv
if ($tasks.Count -eq 0) {
    throw "No tasks found in $TasksCsv"
}

$proc = Get-GeidProcess
$formHandle = Get-GeidFormHandle -ProcessId ([uint32]$proc.Id)
Focus-GeidForm -FormHandle $formHandle

$controls = Get-GeidControls -FormHandle $formHandle
$map = Resolve-GeidControlMap -Controls $controls

Write-Host "Resolved GEID controls:" -ForegroundColor Cyan
$map.GetEnumerator() | ForEach-Object {
    $text = if ($_.Value.PSObject.Properties.Match("Text").Count -gt 0) { $_.Value.Text } else { "" }
    Write-Host ("  {0}: handle={1} text='{2}' top={3} left={4}" -f $_.Key, ('0x{0:X}' -f $_.Value.Handle.ToInt64()), $text, $_.Value.Top, $_.Value.Left)
}

if ((Get-WindowText $map.StartButton.Handle) -eq "Stop") {
    if (-not $ForceStopCurrentTask) {
        throw "GEID is currently running a task. Stop it before filling new parameters, or rerun with -ForceStopCurrentTask."
    }
    Write-Warning "GEID is running a task; clicking Stop because -ForceStopCurrentTask was set."
    Click-Button -Handle $map.StartButton.Handle
    Wait-ForButtonText -Handle $map.StartButton.Handle -ExpectedText "Start" -TimeoutSeconds 30
}

foreach ($task in $tasks) {
    Write-Host ("`n[{0}] Filling task" -f $task.grid_id) -ForegroundColor Green

    Ensure-Directory -PathValue (Split-Path -Parent $task.task_name)
    Ensure-Directory -PathValue $task.save_to

    Set-EditText -Handle $map.TaskName.Handle -Value $task.task_name
    if ($map.Date -and -not [string]::IsNullOrWhiteSpace($task.date)) {
        Set-EditText -Handle $map.Date.Handle -Value $task.date
    }
    Set-EditText -Handle $map.ZoomFrom.Handle -Value ([string]$task.zoom_from)
    Set-EditText -Handle $map.ZoomTo.Handle -Value ([string]$task.zoom_to)
    Set-EditText -Handle $map.LeftLongitude.Handle -Value ([string]$task.left_longitude)
    Set-EditText -Handle $map.RightLongitude.Handle -Value ([string]$task.right_longitude)
    Set-EditText -Handle $map.TopLatitude.Handle -Value ([string]$task.top_latitude)
    Set-EditText -Handle $map.BottomLatitude.Handle -Value ([string]$task.bottom_latitude)
    Set-EditText -Handle $map.SaveTo.Handle -Value $task.save_to

    Write-Host ("  task_name={0}" -f (Get-WindowText $map.TaskName.Handle))
    Write-Host ("  save_to={0}" -f (Get-WindowText $map.SaveTo.Handle))
    Write-Host ("  bounds=({0}, {1}, {2}, {3})" -f
        (Get-WindowText $map.LeftLongitude.Handle),
        (Get-WindowText $map.RightLongitude.Handle),
        (Get-WindowText $map.TopLatitude.Handle),
        (Get-WindowText $map.BottomLatitude.Handle))

    if (-not [string]::IsNullOrWhiteSpace($task.map_type)) {
        Write-Warning "map_type selection is not yet automated; keep GEID combo on '$($task.map_type)' manually."
    }

    if (-not $StartTasks) {
        continue
    }

    Click-Button -Handle $map.StartButton.Handle
    Wait-ForDownload `
        -StartButtonHandle $map.StartButton.Handle `
        -ProcessId ([uint32]$proc.Id) `
        -PollIntervalSeconds $PollSeconds `
        -StartTimeout $StartTimeoutSeconds `
        -CompletionTimeout $CompletionTimeoutSeconds
}
