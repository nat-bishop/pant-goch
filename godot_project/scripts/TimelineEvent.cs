using System.Text.Json.Serialization;

namespace Smallville;

public class TimelineEvent
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("at_seconds")] public int AtSeconds { get; set; }
    [JsonPropertyName("kind")] public string Kind { get; set; } = "";
    [JsonPropertyName("notification_text")] public string NotificationText { get; set; } = "";
}
