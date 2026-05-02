using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Smallville;

/// <summary>
/// Public view of the live handler state. One <see cref="AgentStateView"/>
/// per asset.
/// </summary>
public class HandlerState
{
    [JsonPropertyName("agents")] public List<AgentStateView> Agents { get; set; } = new();
    [JsonPropertyName("actions_remaining")] public int ActionsRemaining { get; set; }
    [JsonPropertyName("max_actions")] public int MaxActions { get; set; }
    [JsonPropertyName("action_count")] public int ActionCount { get; set; }
    [JsonPropertyName("ended")] public bool Ended { get; set; }
    [JsonPropertyName("end_reason")] public string? EndReason { get; set; }
}

public class AgentStateView
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("last_report_action_count")] public int LastReportActionCount { get; set; }
}
