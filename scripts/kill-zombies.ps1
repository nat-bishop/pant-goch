# kill-zombies — clear stale handler/llama/vite processes off the dev ports.
#
# Two-pass kill on Windows:
#   1) Listeners by port (whatever Get-NetTCPConnection reports as OwningProcess).
#   2) Orphan python.exe whose CommandLine mentions `handler` or
#      `multiprocessing-fork`.
#
# Pass 2 exists because uvicorn's multiprocessing.spawn children inherit the
# listening socket handle on Windows — when the parent dies but a child
# survives, the port stays bound to a PID that Get-NetTCPConnection reports as
# dead. Killing the inherited-handle child is the only way to release it.
#
# Verifies ports are free at the end and exits non-zero if any port is still
# bound, so chaining `make kill-zombies run` won't race a half-dead sidecar.

$ports = 8000, 8080, 8081, 5173, 5174, 5175, 5176

Get-NetTCPConnection -LocalPort $ports -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.OwningProcess -gt 0) {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -match 'handler' -or
            $_.CommandLine -match 'multiprocessing-fork'
        )
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Start-Sleep -Milliseconds 500

$still = Get-NetTCPConnection -LocalPort $ports -State Listen -ErrorAction SilentlyContinue
if ($still) {
    Write-Host "kill-zombies FAILED - ports still bound:"
    $still | ForEach-Object {
        Write-Host ("  :{0} PID {1}" -f $_.LocalPort, $_.OwningProcess)
    }
    exit 1
}

Write-Host "kill-zombies: cleared :8000/:8080/:8081/:5173-5176"
