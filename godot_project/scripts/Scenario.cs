using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Smallville;

public class Scenario
{
    [JsonPropertyName("title")] public string Title { get; set; } = "";
    [JsonPropertyName("setup")] public string Setup { get; set; } = "";
    [JsonPropertyName("duration_seconds")] public int DurationSeconds { get; set; }
    [JsonPropertyName("captor")] public Captor Captor { get; set; } = new();
    [JsonPropertyName("hostages")] public List<Hostage> Hostages { get; set; } = new();
}
