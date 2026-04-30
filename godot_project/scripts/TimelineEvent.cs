using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Smallville;

/// <summary>
/// World-clock event surfaced to the client when it fires. <see cref="AffectedAgents"/>
/// names which channels received scene-update injections — used to mark those
/// channels unread and surface the FLASH TRAFFIC banner over the active dossier.
/// </summary>
public class TimelineEvent
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("after_actions")] public int AfterActions { get; set; }
    [JsonPropertyName("kind")] public string Kind { get; set; } = ""; // world_event | flash_traffic | directive
    [JsonPropertyName("notification_text")] public string NotificationText { get; set; } = "";
    [JsonPropertyName("affected_agents")] public List<string> AffectedAgents { get; set; } = new();
}
