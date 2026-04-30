using System.Collections.Generic;
using Godot;

namespace Smallville;

/// <summary>
/// Tag-button dialogue picker. The server's options agent returns 2-5
/// <see cref="DialogueOption"/>s; the panel renders one button per option,
/// labeled with the option's short emotion/intent <c>tag</c>. The full
/// <c>hidden_line</c> is what gets sent to the captor when a button fires.
/// </summary>
public partial class ActionPanel : PanelContainer
{
    [Signal] public delegate void OptionSelectedEventHandler(string hiddenLine);

    private VBoxContainer _root = null!;
    private HBoxContainer _buttonRow = null!;
    private Label _statusLabel = null!;

    private readonly List<Button> _buttons = new();
    private bool _enabled;

    public override void _Ready()
    {
        Build();
    }

    /// <summary>Replace the current button set with a new options list.</summary>
    public void SetOptions(DialogueOption[] options)
    {
        ClearButtons();
        foreach (var opt in options)
        {
            var btn = new Button
            {
                Text = opt.Tag,
                CustomMinimumSize = new Vector2(0, 40),
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                TooltipText = opt.HiddenLine,  // for QA / debug; players generally won't hover
            };
            var hidden = opt.HiddenLine; // capture
            btn.Pressed += () =>
            {
                if (!_enabled) return;
                EmitSignal(SignalName.OptionSelected, hidden);
            };
            _buttonRow.AddChild(btn);
            _buttons.Add(btn);
        }
        _statusLabel.Visible = false;
    }

    /// <summary>Enable or disable all option buttons.</summary>
    public void SetEnabled(bool enabled)
    {
        _enabled = enabled;
        foreach (var b in _buttons) b.Disabled = !enabled;
    }

    public void GrabInputFocus()
    {
        if (_buttons.Count > 0) _buttons[0].GrabFocus();
    }

    public void ShowError(string message)
    {
        ClearButtons();
        _statusLabel.Text = $"Error: {message}";
        _statusLabel.AddThemeColorOverride("font_color", new Color(0.95f, 0.4f, 0.4f));
        _statusLabel.Visible = true;
    }

    /// <summary>Clear buttons and show a placeholder while options are being fetched.</summary>
    public void ShowLoading(string text = "Thinking…")
    {
        ClearButtons();
        _statusLabel.Text = text;
        _statusLabel.AddThemeColorOverride("font_color", new Color(0.65f, 0.65f, 0.65f));
        _statusLabel.Visible = true;
    }

    private void ClearButtons()
    {
        foreach (var b in _buttons)
        {
            b.QueueFree();
        }
        _buttons.Clear();
    }

    private void Build()
    {
        _root = new VBoxContainer
        {
            CustomMinimumSize = new Vector2(0, 80),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        AddChild(_root);

        _buttonRow = new HBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _buttonRow.AddThemeConstantOverride("separation", 8);
        _root.AddChild(_buttonRow);

        _statusLabel = new Label
        {
            Text = "",
            Visible = false,
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        _root.AddChild(_statusLabel);
    }
}
