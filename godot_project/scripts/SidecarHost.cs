using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Godot;

// Disambiguate from Godot.Environment.
using Environment = System.Environment;

namespace Smallville;

/// <summary>
/// Autoload. Spawns the Python sidecar (which in turn spawns llama-server)
/// and tears it down on exit. Single bootstrap point at the game exe.
/// </summary>
public partial class SidecarHost : Node
{
    private Process? _proc;
    private bool _skipSpawn;
    // Windows Job Object handle. Held for the lifetime of this process so that
    // when Godot dies for ANY reason (clean exit, Ctrl-C in console, crash,
    // taskkill), the OS closes the handle and kills every member process —
    // Python sidecar AND its grandchild llama-server. Without this, Ctrl-C in
    // the terminal often skips _ExitTree() and leaves :8000 / :8080 bound.
    private IntPtr _jobHandle = IntPtr.Zero;

    public override void _Ready()
    {
        var args = OS.GetCmdlineArgs();
        _skipSpawn = args.Contains("--no-sidecar");

        if (_skipSpawn)
        {
            GD.Print("[SidecarHost] --no-sidecar — skipping spawn (assume sidecar managed externally)");
            return;
        }

        try
        {
            EnsureJobObject();
            SpawnSidecar();
        }
        catch (Exception e)
        {
            GD.PrintErr($"[SidecarHost] failed to spawn Python sidecar: {e}");
        }
    }

    public override void _ExitTree()
    {
        if (_proc == null || _proc.HasExited)
        {
            CloseJobHandle();
            return;
        }
        try
        {
            GD.Print($"[SidecarHost] killing sidecar pid={_proc.Id} (entire process tree)");
            _proc.Kill(entireProcessTree: true);
            _proc.WaitForExit(5000);
        }
        catch (Exception e)
        {
            GD.PrintErr($"[SidecarHost] shutdown failed: {e}");
        }
        finally
        {
            CloseJobHandle();
        }
    }

    private void SpawnSidecar()
    {
        string exeDir = OS.GetExecutablePath().GetBaseDir();
        string projectRoot = ProjectSettings.GlobalizePath("res://");

        string sidecarDir = ResolveSidecarDir(exeDir, projectRoot);
        string python = ResolvePython(exeDir);

        var psi = new ProcessStartInfo
        {
            FileName = python,
            // Run from the npc_service project root so the .venv at
            // npc_service/.venv is the active environment. The package itself
            // lives at npc_service/src/npc_service and is installed editable
            // by `uv sync`, so PYTHONPATH is not needed.
            WorkingDirectory = sidecarDir,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        // python -m npc_service
        psi.ArgumentList.Add("-m");
        psi.ArgumentList.Add("npc_service");

        // Force UTF-8 stdio so logs render emoji / em-dashes correctly on Windows.
        psi.Environment["PYTHONUTF8"] = "1";

        GD.Print($"[SidecarHost] spawning: {python} -m npc_service (cwd={sidecarDir})");

        _proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _proc.OutputDataReceived += (_, e) =>
        {
            if (e.Data != null) GD.Print($"[npc] {e.Data}");
        };
        _proc.ErrorDataReceived += (_, e) =>
        {
            if (e.Data != null) GD.PrintErr($"[npc] {e.Data}");
        };
        _proc.Exited += (_, _) =>
        {
            GD.PrintErr($"[SidecarHost] sidecar exited unexpectedly (rc={_proc?.ExitCode})");
        };

        _proc.Start();
        // Assign before draining stdout/err so a fast-failing child can't spawn
        // and detach a grandchild before we get the chance to enroll it.
        AssignProcessToJob(_proc);
        _proc.BeginOutputReadLine();
        _proc.BeginErrorReadLine();
    }

    // ───────────────────────── Win32 Job Object plumbing ─────────────────────────
    //
    // Why: on Windows, Ctrl-C in the terminal that ran `make run` does not always
    // run autoload _ExitTree() — Godot is a GUI app and the console signal can
    // skip clean teardown. A Job Object with KILL_ON_JOB_CLOSE is the canonical
    // OS-level guarantee: when the handle held by this .NET process is closed
    // (by clean code below OR by the kernel when Godot dies abruptly), every
    // process in the job is terminated. Children inherit job membership by
    // default, so llama-server (Python's child) is covered too.

    private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;
    private const int JobObjectExtendedLimitInformation = 9;

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObjectW(IntPtr lpJobAttributes, string? lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(
        IntPtr hJob,
        int JobObjectInformationClass,
        ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION lpJobObjectInformation,
        uint cbJobObjectInformationLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CloseHandle(IntPtr hObject);

    private void EnsureJobObject()
    {
        if (!OS.HasFeature("windows")) return;
        if (_jobHandle != IntPtr.Zero) return;

        IntPtr h = CreateJobObjectW(IntPtr.Zero, null);
        if (h == IntPtr.Zero)
        {
            GD.PrintErr($"[SidecarHost] CreateJobObject failed: {Marshal.GetLastWin32Error()} — child cleanup falls back to _ExitTree()");
            return;
        }

        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(
                h,
                JobObjectExtendedLimitInformation,
                ref info,
                (uint)Marshal.SizeOf<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>()))
        {
            GD.PrintErr($"[SidecarHost] SetInformationJobObject failed: {Marshal.GetLastWin32Error()}");
            CloseHandle(h);
            return;
        }

        _jobHandle = h;
        GD.Print("[SidecarHost] job object armed (KILL_ON_JOB_CLOSE) — children will die with Godot");
    }

    private void AssignProcessToJob(Process proc)
    {
        if (!OS.HasFeature("windows")) return;
        if (_jobHandle == IntPtr.Zero) return;
        if (!AssignProcessToJobObject(_jobHandle, proc.Handle))
        {
            GD.PrintErr($"[SidecarHost] AssignProcessToJobObject failed: {Marshal.GetLastWin32Error()}");
        }
    }

    private void CloseJobHandle()
    {
        if (_jobHandle == IntPtr.Zero) return;
        // Closing the last handle to the job triggers KILL_ON_JOB_CLOSE.
        CloseHandle(_jobHandle);
        _jobHandle = IntPtr.Zero;
    }

    /// <summary>
    /// Three-tier Python resolve: env override -> bundled (ship) -> PATH (dev).
    /// </summary>
    private static string ResolvePython(string exeDir)
    {
        var env = Environment.GetEnvironmentVariable("NPC_PYTHON");
        if (!string.IsNullOrEmpty(env) && File.Exists(env))
        {
            return env;
        }
        string bundled = Path.Combine(exeDir, "python", OS.HasFeature("windows") ? "python.exe" : "python");
        if (File.Exists(bundled))
        {
            return bundled;
        }
        // Dev: rely on PATH. Try to find a .venv in the sidecar dir first so we use uv-managed deps.
        string projectRoot = ProjectSettings.GlobalizePath("res://");
        string sidecarDir = ResolveSidecarDir(exeDir, projectRoot);
        string venvPython = Path.Combine(
            sidecarDir, ".venv",
            OS.HasFeature("windows") ? "Scripts" : "bin",
            OS.HasFeature("windows") ? "python.exe" : "python");
        if (File.Exists(venvPython))
        {
            return venvPython;
        }
        return OS.HasFeature("windows") ? "python.exe" : "python";
    }

    /// <summary>
    /// Resolve the npc_service directory in dev or ship layouts.
    /// Dev: project sits at &lt;repo&gt;/godot_project/, sidecar at &lt;repo&gt;/npc_service/.
    /// Ship: both at &lt;exeDir&gt;/.
    /// </summary>
    private static string ResolveSidecarDir(string exeDir, string projectRoot)
    {
        var env = Environment.GetEnvironmentVariable("NPC_SERVICE_DIR");
        if (!string.IsNullOrEmpty(env) && Directory.Exists(env))
        {
            return env;
        }
        // Ship: <exeDir>/npc_service
        string ship = Path.Combine(exeDir, "npc_service");
        if (Directory.Exists(ship))
        {
            return ship;
        }
        // Dev: <projectRoot>/../npc_service
        string dev = Path.GetFullPath(Path.Combine(projectRoot, "..", "npc_service"));
        return dev;
    }
}
