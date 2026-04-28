namespace Smallville;

public enum ChatMessageRole
{
    PlayerAction,
    Captor,
    System,
}

public class ChatMessage
{
    public ChatMessageRole Role { get; init; }
    public string Text { get; set; } = "";
    public bool Interrupted { get; set; } = false;
}
