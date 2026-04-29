using System.Text.Json.Serialization;

namespace Smallville;

/// <summary>
/// One generated dialogue option the player can pick.
/// `Line` is shown verbatim (button label + chat log entry) and sent verbatim
/// to the captor. `ActionType` and `Detail` are hidden tags from the option
/// generator — they ride along on /action so the server can log them.
/// Mirrors npc_service.models.PlayerOption.
/// </summary>
public class PlayerOption
{
    [JsonPropertyName("line")] public string Line { get; set; } = "";
    [JsonPropertyName("action_type")] public string ActionType { get; set; } = "";
    [JsonPropertyName("detail")] public string Detail { get; set; } = "";
}

public class OptionsResponse
{
    [JsonPropertyName("options")] public PlayerOption[] Options { get; set; } = System.Array.Empty<PlayerOption>();
}
