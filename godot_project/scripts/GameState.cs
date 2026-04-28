using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Smallville;

public class GameState
{
    [JsonPropertyName("hostages")] public List<Hostage> Hostages { get; set; } = new();
    [JsonPropertyName("time_remaining_seconds")] public int TimeRemainingSeconds { get; set; }
    [JsonPropertyName("ended")] public bool Ended { get; set; }
    [JsonPropertyName("captor_outcome")] public string? CaptorOutcome { get; set; }
    [JsonPropertyName("end_reason")] public string? EndReason { get; set; }
}
