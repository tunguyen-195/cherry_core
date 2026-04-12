$ModelDir = "E:\research\Cherry2\cherry_core"
$WSLUser = "root" 

Write-Host "🍒 Cherry Core V2 - WSL2 vLLM Setup Automation" -ForegroundColor Cyan

# 1. Check WSL Status
$params = @{
    CommandLine = "wsl --status"
    StdOut = $null
    StdErr = $null
}

try {
    wsl --status | Out-Null
    Write-Host "✅ WSL2 is active." -ForegroundColor Green
} catch {
    Write-Host "❌ WSL2 is NOT running. Please run 'wsl --install' and restart your PC." -ForegroundColor Red
    exit
}

$WslPath = "/mnt/e/research/Cherry2/cherry_core"

# 3. Execute inside WSL
# Convert Windows path E:\... to /mnt/e/... for the execution
# We assume the user runs this from the scripts folder or root. 
# We'll use relative paths if running from scripts folder.

Write-Host "⏳ Executing setup inside WSL (Ubuntu)... This may take 5-10 minutes." -ForegroundColor Yellow
wsl -e bash $WslPath/scripts/wsl_setup.sh

Write-Host "🎉 Finished!" -ForegroundColor Green
Write-Host "👉 Running Verification Test in WSL..." -ForegroundColor Cyan

wsl -e bash $WslPath/scripts/wsl_verify.sh

Write-Host "✅ Done."
