using Godot;

namespace Smallville;

/// <summary>
/// Free-text dialogue input. Player types a line and submits with Enter or
/// the Send button; emits ActionSelected(line) which Game.cs forwards to
/// /action.
/// </summary>
public partial class ActionPanel : PanelContainer
{
    [Signal] public delegate void ActionSelectedEventHandler(string line);

    private LineEdit _input = null!;
    private Button _sendButton = null!;
    private Label _statusLabel = null!;

    public override void _Ready()
    {
        Build();
    }

    public void SetEnabled(bool enabled)
    {
        _input.Editable = enabled;
        _sendButton.Disabled = !enabled;
        if (enabled) _statusLabel.Visible = false;
    }

    public void GrabInputFocus()
    {
        _input.GrabFocus();
    }

    public void ShowError(string message)
    {
        _statusLabel.Text = $"Error: {message}";
        _statusLabel.AddThemeColorOverride("font_color", new Color(0.95f, 0.4f, 0.4f));
        _statusLabel.Visible = true;
    }

    private void Submit()
    {
        if (!_input.Editable) return;
        var line = _input.Text.Trim();
        if (line.Length == 0) return;
        _input.Text = "";
        EmitSignal(SignalName.ActionSelected, line);
    }

    private void OnTextSubmitted(string _) => Submit();
    private void OnSendPressed() => Submit();

    private void Build()
    {
        var root = new VBoxContainer
        {
            CustomMinimumSize = new Vector2(0, 80),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        AddChild(root);

        var inputRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        root.AddChild(inputRow);

        _input = new LineEdit
        {
            PlaceholderText = "What do you say?",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0, 40),
        };
        _input.TextSubmitted += OnTextSubmitted;
        inputRow.AddChild(_input);

        _sendButton = new Button
        {
            Text = "Send",
            CustomMinimumSize = new Vector2(80, 40),
        };
        _sendButton.Pressed += OnSendPressed;
        inputRow.AddChild(_sendButton);

        _statusLabel = new Label
        {
            Text = "",
            Visible = false,
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        root.AddChild(_statusLabel);
    }
}
