using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Smallville;

public class GameState
{
    [JsonPropertyName("hostages")] public List<Hostage> Hostages { get; set; } = new();
    [JsonPropertyName("actions_remaining")] public int ActionsRemaining { get; set; }
    [JsonPropertyName("max_actions")] public int MaxActions { get; set; }
    [JsonPropertyName("ended")] public bool Ended { get; set; }
    [JsonPropertyName("captor_outcome")] public string? CaptorOutcome { get; set; }
    [JsonPropertyName("end_reason")] public string? EndReason { get; set; }

    /// <summary>Populated only when the server runs with DEBUG_AFFORDANCES=true.</summary>
    [JsonPropertyName("debug")] public DebugView? Debug { get; set; }
}

/// <summary>Server-side hidden state, exposed only with DEBUG_AFFORDANCES=true.</summary>
public class DebugView
{
    [JsonPropertyName("stress")] public int Stress { get; set; }
    [JsonPropertyName("trust")] public int Trust { get; set; }
    [JsonPropertyName("desperation")] public int Desperation { get; set; }
    [JsonPropertyName("action_count")] public int ActionCount { get; set; }
    [JsonPropertyName("max_actions")] public int MaxActions { get; set; }
    [JsonPropertyName("fired_timeline_event_ids")] public List<string> FiredTimelineEventIds { get; set; } = new();
}
