using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Smallville;

public class Captor
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("persona")] public string Persona { get; set; } = "";
    [JsonPropertyName("initial_demands")] public List<string> InitialDemands { get; set; } = new();
}
