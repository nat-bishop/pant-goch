using System;
using System.IO;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;

// Disambiguate from Godot.HttpClient / Godot.Environment.
using HttpClient = System.Net.Http.HttpClient;
using HttpRequestMessage = System.Net.Http.HttpRequestMessage;
using HttpMethod = System.Net.Http.HttpMethod;
using HttpCompletionOption = System.Net.Http.HttpCompletionOption;
using Environment = System.Environment;

namespace Smallville;

/// <summary>
/// Autoload. Wraps HttpClient calls to the Python sidecar. Streams /action
/// and /events via SSE. All Godot-side callbacks marshalled to the main
/// thread via Callable.CallDeferred.
/// </summary>
public partial class LLMClient : Node
{
    [Signal] public delegate void SidecarReadyEventHandler();

    private static readonly HttpClient _http = new()
    {
        // SSE streaming requires no default timeout. Per-request CancellationToken instead.
        Timeout = Timeout.InfiniteTimeSpan,
    };

    public string BaseUrl { get; private set; } = "http://127.0.0.1:8000";

    public override void _Ready()
    {
        var env = Environment.GetEnvironmentVariable("NPC_SERVICE_URL");
        if (!string.IsNullOrEmpty(env))
        {
            BaseUrl = env.TrimEnd('/');
        }
        GD.Print($"[LLMClient] base url = {BaseUrl}");
    }

    /// <summary>
    /// Poll /health every 500 ms until 200, or timeout. Emits Ready signal on success.
    /// </summary>
    public async Task WaitReady(double timeoutSeconds = 60.0, CancellationToken ct = default)
    {
        var deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
        Exception? last = null;
        while (DateTime.UtcNow < deadline && !ct.IsCancellationRequested)
        {
            try
            {
                using var resp = await _http.GetAsync($"{BaseUrl}/health", ct);
                if (resp.IsSuccessStatusCode)
                {
                    GD.Print("[LLMClient] sidecar healthy");
                    Callable.From(() => EmitSignal(SignalName.SidecarReady)).CallDeferred();
                    return;
                }
            }
            catch (Exception e)
            {
                last = e;
            }
            try { await Task.Delay(500, ct); } catch (TaskCanceledException) { break; }
        }
        throw new TimeoutException(
            $"sidecar at {BaseUrl}/health did not respond within {timeoutSeconds}s. Last error: {last}");
    }

    public async Task<Scenario> GetScenarioAsync(CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync($"{BaseUrl}/scenario", ct);
        resp.EnsureSuccessStatusCode();
        var json = await resp.Content.ReadAsStringAsync(ct);
        return JsonSerializer.Deserialize<Scenario>(json) ?? new Scenario();
    }

    public async Task<GameState> GetStateAsync(CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync($"{BaseUrl}/state", ct);
        resp.EnsureSuccessStatusCode();
        var json = await resp.Content.ReadAsStringAsync(ct);
        return JsonSerializer.Deserialize<GameState>(json) ?? new GameState();
    }

    public async Task<GameState> ResetAsync(CancellationToken ct = default)
    {
        using var resp = await _http.PostAsync($"{BaseUrl}/reset", content: null, ct);
        resp.EnsureSuccessStatusCode();
        var json = await resp.Content.ReadAsStringAsync(ct);
        return JsonSerializer.Deserialize<GameState>(json) ?? new GameState();
    }

    /// <summary>
    /// POST /action with the player's free-text line and stream the captor's
    /// structured response. Text deltas, then exactly one state event, then done.
    /// </summary>
    public Task StreamActionAsync(
        string line,
        Action<string> onTextDelta,
        Action<GameState> onState,
        Action onDone,
        Action<string> onError,
        CancellationToken ct = default)
        => StreamSseAsync(
            new HttpRequestMessage(HttpMethod.Post, $"{BaseUrl}/action")
            {
                Content = JsonContent.Create(new { line }),
            },
            onTextDelta, onState, onDone, onError, ct);

    /// <summary>
    /// POST /regenerate. Pops the last exchange server-side and re-streams.
    /// Server enforces DEBUG_AFFORDANCES=true.
    /// </summary>
    public Task RegenerateAsync(
        Action<string> onTextDelta,
        Action<GameState> onState,
        Action onDone,
        Action<string> onError,
        CancellationToken ct = default)
        => StreamSseAsync(
            new HttpRequestMessage(HttpMethod.Post, $"{BaseUrl}/regenerate"),
            onTextDelta, onState, onDone, onError, ct);

    /// <summary>
    /// Subscribe to /events SSE — timeline-event notifications and game-end signals.
    /// Lifetime = whole game session. Returns a Task you can cancel via the token.
    /// </summary>
    public async Task SubscribeEventsAsync(
        Action<TimelineEvent> onTimelineEvent,
        Action<string, string> onGameEnded, // (endReason, captorOutcome)
        Action<string> onError,
        CancellationToken ct = default)
    {
        try
        {
            var req = new HttpRequestMessage(HttpMethod.Get, $"{BaseUrl}/events");
            using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
            resp.EnsureSuccessStatusCode();
            using var stream = await resp.Content.ReadAsStreamAsync(ct);
            using var reader = new StreamReader(stream);
            string? line;
            while ((line = await reader.ReadLineAsync(ct)) != null)
            {
                if (line.Length == 0 || line.StartsWith(":")) continue;
                if (!line.StartsWith("data: ")) continue;
                var payload = line.Substring(6);
                using var doc = JsonDocument.Parse(payload);
                var type = doc.RootElement.GetProperty("type").GetString();
                switch (type)
                {
                    case "timeline_event":
                    {
                        var evJson = doc.RootElement.GetProperty("event").GetRawText();
                        var ev = JsonSerializer.Deserialize<TimelineEvent>(evJson);
                        if (ev != null)
                        {
                            Callable.From(() => onTimelineEvent(ev)).CallDeferred();
                        }
                        break;
                    }
                    case "game_ended":
                    {
                        var er = doc.RootElement.TryGetProperty("end_reason", out var e1) ? e1.GetString() ?? "" : "";
                        var co = doc.RootElement.TryGetProperty("captor_outcome", out var e2) ? e2.GetString() ?? "" : "";
                        Callable.From(() => onGameEnded(er, co)).CallDeferred();
                        break;
                    }
                    case "reset":
                    case "connected":
                    default:
                        // ignore
                        break;
                }
            }
        }
        catch (OperationCanceledException) { /* expected on game shutdown */ }
        catch (Exception e)
        {
            Callable.From(() => onError(e.Message)).CallDeferred();
        }
    }

    private async Task StreamSseAsync(
        HttpRequestMessage req,
        Action<string> onTextDelta,
        Action<GameState> onState,
        Action onDone,
        Action<string> onError,
        CancellationToken ct)
    {
        try
        {
            // CRITICAL: ResponseHeadersRead — without this HttpClient buffers
            // the entire body before returning, breaking SSE.
            using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
            resp.EnsureSuccessStatusCode();
            using var stream = await resp.Content.ReadAsStreamAsync(ct);
            using var reader = new StreamReader(stream);
            string? line;
            while ((line = await reader.ReadLineAsync(ct)) != null)
            {
                if (line.Length == 0 || line.StartsWith(":")) continue;
                if (!line.StartsWith("data: ")) continue;
                var payload = line.Substring(6);
                using var doc = JsonDocument.Parse(payload);
                var type = doc.RootElement.GetProperty("type").GetString();
                switch (type)
                {
                    case "text":
                    {
                        var t = doc.RootElement.GetProperty("text").GetString() ?? "";
                        Callable.From(() => onTextDelta(t)).CallDeferred();
                        break;
                    }
                    case "state":
                    {
                        var sj = doc.RootElement.GetProperty("state").GetRawText();
                        var st = JsonSerializer.Deserialize<GameState>(sj) ?? new GameState();
                        Callable.From(() => onState(st)).CallDeferred();
                        break;
                    }
                    case "done":
                        Callable.From(() => onDone()).CallDeferred();
                        return;
                    case "error":
                    {
                        var msg = doc.RootElement.GetProperty("message").GetString() ?? "";
                        Callable.From(() => onError(msg)).CallDeferred();
                        return;
                    }
                }
            }
            // stream ended without explicit done — treat as done
            Callable.From(() => onDone()).CallDeferred();
        }
        catch (OperationCanceledException)
        {
            // Cancelled by caller (player switched action / pressed stop).
            // No callback — caller knows it cancelled.
        }
        catch (Exception e)
        {
            GD.PrintErr($"[LLMClient] stream failed: {e}");
            Callable.From(() => onError(e.Message)).CallDeferred();
        }
        finally
        {
            req.Dispose();
        }
    }
}
