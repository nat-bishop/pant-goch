using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Smallville;

/// <summary>
/// Public view of the live handler state. One <see cref="AgentStateView"/>
/// per asset (no per-agent meters in the public view; meters appear under
/// <see cref="Debug"/> only when DEBUG_AFFORDANCES=true on the server).
/// </summary>
public class HandlerState
{
    [JsonPropertyName("agents")] public List<AgentStateView> Agents { get; set; } = new();
    [JsonPropertyName("actions_remaining")] public int ActionsRemaining { get; set; }
    [JsonPropertyName("max_actions")] public int MaxActions { get; set; }
    [JsonPropertyName("action_count")] public int ActionCount { get; set; }
    [JsonPropertyName("ended")] public bool Ended { get; set; }
    [JsonPropertyName("end_reason")] public string? EndReason { get; set; }

    [JsonPropertyName("debug")] public DebugView? Debug { get; set; }
}

public class AgentStateView
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("callsign")] public string Callsign { get; set; } = "";
    [JsonPropertyName("city")] public string City { get; set; } = "";
    [JsonPropertyName("lost")] public bool Lost { get; set; }
    [JsonPropertyName("last_report_action_count")] public int LastReportActionCount { get; set; }
}

/// <summary>
/// Server-side hidden state envelope, exposed only with DEBUG_AFFORDANCES=true.
/// One <see cref="AgentDebugView"/> per asset.
/// </summary>
public class DebugView
{
    [JsonPropertyName("agents")] public List<AgentDebugView> Agents { get; set; } = new();
    [JsonPropertyName("action_count")] public int ActionCount { get; set; }
    [JsonPropertyName("max_actions")] public int MaxActions { get; set; }
    [JsonPropertyName("fired_timeline_event_ids")] public List<string> FiredTimelineEventIds { get; set; } = new();
}

public class AgentDebugView
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("stress")] public int Stress { get; set; }
    [JsonPropertyName("trust")] public int Trust { get; set; }
}
