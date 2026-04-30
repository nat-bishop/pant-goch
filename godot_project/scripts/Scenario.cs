using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Smallville;

public class Scenario
{
    [JsonPropertyName("title")] public string Title { get; set; } = "";
    [JsonPropertyName("setup")] public string Setup { get; set; } = "";
    [JsonPropertyName("max_actions")] public int MaxActions { get; set; }
    [JsonPropertyName("captor")] public Captor Captor { get; set; } = new();
    [JsonPropertyName("hostages")] public List<Hostage> Hostages { get; set; } = new();
}
