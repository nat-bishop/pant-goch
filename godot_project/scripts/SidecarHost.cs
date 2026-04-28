using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
    }

    private void SpawnSidecar()
    {
        string exeDir = OS.GetExecutablePath().GetBaseDir();
        string projectRoot = ProjectSettings.GlobalizePath("res://");

        string sidecarDir = ResolveSidecarDir(exeDir, projectRoot);
        string? maybeParent = Path.GetDirectoryName(sidecarDir.TrimEnd('/', '\\'));
        string parentOfSidecar = maybeParent ?? sidecarDir;
        string python = ResolvePython(exeDir);

        var psi = new ProcessStartInfo
        {
            FileName = python,
            WorkingDirectory = parentOfSidecar,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        // python -m npc_service
        psi.ArgumentList.Add("-m");
        psi.ArgumentList.Add("npc_service");

        // Forward important env vars; ensure parent of npc_service is on PYTHONPATH
        // so `python -m npc_service` resolves the package even when uv didn't install it.
        var existingPp = Environment.GetEnvironmentVariable("PYTHONPATH") ?? "";
        psi.Environment["PYTHONPATH"] = string.IsNullOrEmpty(existingPp)
            ? parentOfSidecar
            : parentOfSidecar + Path.PathSeparator + existingPp;
        // Force UTF-8 stdio so logs render emoji / em-dashes correctly on Windows.
        psi.Environment["PYTHONUTF8"] = "1";

        GD.Print($"[SidecarHost] spawning: {python} -m npc_service (cwd={parentOfSidecar})");

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
        _proc.BeginOutputReadLine();
        _proc.BeginErrorReadLine();
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
