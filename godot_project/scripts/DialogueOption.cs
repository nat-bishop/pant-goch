using System.Text.Json.Serialization;

namespace Smallville;

/// <summary>
/// One LLM-generated dialogue pick. The player sees only <see cref="Tag"/>
/// (a short emotion/intent label like "Empathize"); <see cref="HiddenLine"/>
/// is the full sentence sent to the captor when this option is chosen.
/// </summary>
public class DialogueOption
{
    [JsonPropertyName("tag")] public string Tag { get; set; } = "";
    [JsonPropertyName("hidden_line")] public string HiddenLine { get; set; } = "";
}
