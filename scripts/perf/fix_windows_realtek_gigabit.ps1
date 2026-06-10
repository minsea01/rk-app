#requires -RunAsAdministrator
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $env:USERPROFILE "Desktop\rk3588_45ms_latency_diagnosis\windows_net"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$ip = Get-NetIPAddress -IPAddress "192.168.137.1" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ip) {
    $adapter = Get-NetAdapter -InterfaceIndex $ip.InterfaceIndex
} else {
    $adapter = Get-NetAdapter | Where-Object { $_.InterfaceDescription -like "*Realtek PCIe GbE*" } | Select-Object -First 1
}

if (-not $adapter) {
    throw "Realtek PCIe GbE adapter or 192.168.137.1 adapter was not found."
}

$beforePath = Join-Path $outDir "realtek_before_admin_fix_$stamp.json"
$afterPath = Join-Path $outDir "realtek_after_admin_fix_$stamp.json"

$beforeProps = Get-NetAdapterAdvancedProperty -Name $adapter.Name
[pscustomobject]@{
    adapter = $adapter | Select-Object Name, InterfaceDescription, Status, LinkSpeed, MacAddress
    advanced = $beforeProps | Select-Object DisplayName, DisplayValue, RegistryKeyword, RegistryValue
} | ConvertTo-Json -Depth 6 | Out-File -Encoding UTF8 $beforePath

$settings = @(
    @{ Keyword = "GigaLite"; Value = 0 },
    @{ Keyword = "EnableGreenEthernet"; Value = 0 },
    @{ Keyword = "*EEE"; Value = 0 },
    @{ Keyword = "AdvancedEEE"; Value = 0 },
    @{ Keyword = "AutoDisableGigabit"; Value = 0 },
    @{ Keyword = "*SpeedDuplex"; Value = 0 }
)

$changes = @()
foreach ($setting in $settings) {
    try {
        Set-NetAdapterAdvancedProperty -Name $adapter.Name `
            -RegistryKeyword $setting.Keyword `
            -RegistryValue $setting.Value `
            -NoRestart `
            -ErrorAction Stop
        $changes += [pscustomobject]@{
            RegistryKeyword = $setting.Keyword
            RegistryValue = $setting.Value
            Status = "OK"
            Error = ""
        }
    } catch {
        $changes += [pscustomobject]@{
            RegistryKeyword = $setting.Keyword
            RegistryValue = $setting.Value
            Status = "SKIPPED_OR_FAILED"
            Error = $_.Exception.Message
        }
    }
}

Restart-NetAdapter -Name $adapter.Name -Confirm:$false
Start-Sleep -Seconds 8

$adapterAfter = Get-NetAdapter -Name $adapter.Name
$afterProps = Get-NetAdapterAdvancedProperty -Name $adapter.Name
[pscustomobject]@{
    changes = $changes
    adapter = $adapterAfter | Select-Object Name, InterfaceDescription, Status, LinkSpeed, MacAddress
    advanced = $afterProps | Select-Object DisplayName, DisplayValue, RegistryKeyword, RegistryValue
} | ConvertTo-Json -Depth 6 | Out-File -Encoding UTF8 $afterPath

Write-Host "Saved before: $beforePath"
Write-Host "Saved after:  $afterPath"
Write-Host "Current adapter state:"
$adapterAfter | Select-Object Name, InterfaceDescription, Status, LinkSpeed, MacAddress | Format-List
