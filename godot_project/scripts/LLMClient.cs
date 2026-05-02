using System;
using System.IO;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
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
///
/// Per-channel routing: /action and /options are now keyed by agent_id; SSE
/// events from /action carry agent_id so the client can route into the right
/// channel transcript even if the player has switched dossiers mid-stream.
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

    public async Task<HandlerState> GetStateAsync(CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync($"{BaseUrl}/state", ct);
        resp.EnsureSuccessStatusCode();
        var json = await resp.Content.ReadAsStringAsync(ct);
        return JsonSerializer.Deserialize<HandlerState>(json) ?? new HandlerState();
    }

    public async Task<HandlerState> ResetAsync(CancellationToken ct = default)
    {
        using var resp = await _http.PostAsync($"{BaseUrl}/reset", content: null, ct);
        resp.EnsureSuccessStatusCode();
        var json = await resp.Content.ReadAsStringAsync(ct);
        return JsonSerializer.Deserialize<HandlerState>(json) ?? new HandlerState();
    }

    /// <summary>
    /// GET /options?agent_id=… — fetch 2-5 LLM-generated dialogue picks for
    /// the handler's next line in the named channel. Returns empty array if
    /// the game has ended.
    /// </summary>
    public async Task<DialogueOption[]> GetOptionsAsync(string agentId, CancellationToken ct = default)
    {
        var url = $"{BaseUrl}/options?agent_id={Uri.EscapeDataString(agentId)}";
        using var resp = await _http.GetAsync(url, ct);
        resp.EnsureSuccessStatusCode();
        var json = await resp.Content.ReadAsStringAsync(ct);
        var doc = JsonSerializer.Deserialize<OptionsEnvelope>(json);
        return doc?.Options ?? Array.Empty<DialogueOption>();
    }

    private class OptionsEnvelope
    {
        [JsonPropertyName("options")] public DialogueOption[] Options { get; set; } = Array.Empty<DialogueOption>();
    }

    /// <summary>
    /// POST /action {agent_id, line} and stream the agent's structured response.
    /// SSE events: text deltas (with agent_id), one state event, optional
    /// timeline events (action-counted, fired inline), then done.
    /// </summary>
    public Task StreamActionAsync(
        string agentId,
        string line,
        Action<string, string> onTextDelta, // (agentId, text delta)
        Action<HandlerState> onState,
        Action<TimelineEvent> onTimeline,
        Action onDone,
        Action<string> onError,
        CancellationToken ct = default)
        => StreamSseAsync(
            new HttpRequestMessage(HttpMethod.Post, $"{BaseUrl}/action")
            {
                Content = JsonContent.Create(new { agent_id = agentId, line }),
            },
            onTextDelta, onState, onTimeline, onDone, onError, ct);

    /// <summary>
    /// Subscribe to /events SSE — game-end and reset signals. Timeline events
    /// flow inline through /action SSE, not here. Lifetime = whole game session.
    /// </summary>
    public async Task SubscribeEventsAsync(
        Action<string> onGameEnded, // (endReason)
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
                    case "game_ended":
                    {
                        var er = doc.RootElement.TryGetProperty("end_reason", out var e1) ? e1.GetString() ?? "" : "";
                        Callable.From(() => onGameEnded(er)).CallDeferred();
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
        Action<string, string> onTextDelta,
        Action<HandlerState> onState,
        Action<TimelineEvent> onTimeline,
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
                        var aid = doc.RootElement.TryGetProperty("agent_id", out var a) ? a.GetString() ?? "" : "";
                        Callable.From(() => onTextDelta(aid, t)).CallDeferred();
                        break;
                    }
                    case "state":
                    {
                        var sj = doc.RootElement.GetProperty("state").GetRawText();
                        var st = JsonSerializer.Deserialize<HandlerState>(sj) ?? new HandlerState();
                        Callable.From(() => onState(st)).CallDeferred();
                        break;
                    }
                    case "timeline":
                    {
                        var evJson = doc.RootElement.GetProperty("event").GetRawText();
                        var ev = JsonSerializer.Deserialize<TimelineEvent>(evJson);
                        if (ev != null)
                        {
                            Callable.From(() => onTimeline(ev)).CallDeferred();
                        }
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
