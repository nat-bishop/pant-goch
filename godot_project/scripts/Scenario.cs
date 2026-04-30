using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Smallville;

public class Scenario
{
    [JsonPropertyName("title")] public string Title { get; set; } = "";
    [JsonPropertyName("setup")] public string Setup { get; set; } = "";
    [JsonPropertyName("max_actions")] public int MaxActions { get; set; }
    [JsonPropertyName("operation")] public Operation Operation { get; set; } = new();
    [JsonPropertyName("agents")] public List<AgentDef> Agents { get; set; } = new();
}
