using System.Text.Json.Serialization;

namespace Smallville;

public class Hostage
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("description")] public string Description { get; set; } = "";
    [JsonPropertyName("status")] public string Status { get; set; } = "held"; // held | released | harmed
}
