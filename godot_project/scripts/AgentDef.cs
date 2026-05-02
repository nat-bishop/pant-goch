using System.Text.Json.Serialization;

namespace Smallville;

/// <summary>
/// Public view of one field asset, returned by /scenario. Server omits
/// known_facts, seed_exchanges, deflection_style.
/// <see cref="OpeningTransmission"/> is the agent's first line, rendered into
/// the chat at boot — not a player-typed message.
/// </summary>
public class AgentDef
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("callsign")] public string Callsign { get; set; } = "";
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("city")] public string City { get; set; } = "";
    [JsonPropertyName("role")] public string Role { get; set; } = "";
    [JsonPropertyName("persona")] public string Persona { get; set; } = "";
    [JsonPropertyName("opening_transmission")] public string OpeningTransmission { get; set; } = "";
}
