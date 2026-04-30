using System.Text.Json.Serialization;

namespace Smallville;

/// <summary>
/// Operation brief. Server omits world_state from the public view.
/// </summary>
public class Operation
{
    [JsonPropertyName("codename")] public string Codename { get; set; } = "";
    [JsonPropertyName("objective")] public string Objective { get; set; } = "";
}
