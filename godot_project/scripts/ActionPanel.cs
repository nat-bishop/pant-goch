using Godot;

namespace Smallville;

/// <summary>
/// Dynamic dialogue-option UI. Each turn the panel is fed 3 PlayerOption
/// objects from the sidecar's /options endpoint and renders one button per
/// line. On press it emits ActionSelected(type, detail, line) — type and
/// detail come from the option's hidden tags.
/// </summary>
public partial class ActionPanel : PanelContainer
{
    [Signal] public delegate void ActionSelectedEventHandler(string type, string detail, string line);

    private VBoxContainer _root = null!;
    private Label _header = null!;
    private VBoxContainer _buttonsBox = null!;
    private Label _statusLabel = null!;
    private bool _enabled = false;

    public override void _Ready()
    {
        Build();
    }

    public void SetEnabled(bool enabled)
    {
        _enabled = enabled;
        ApplyEnabled(_buttonsBox);
    }

    public void ShowLoading()
    {
        ClearButtons();
        _statusLabel.Text = "Generating options…";
        _statusLabel.RemoveThemeColorOverride("font_color");
        _statusLabel.Visible = true;
        _enabled = false;
    }

    public void ShowError(string message)
    {
        ClearButtons();
        _statusLabel.Text = $"Options error: {message}";
        _statusLabel.AddThemeColorOverride("font_color", new Color(0.95f, 0.4f, 0.4f));
        _statusLabel.Visible = true;
        _enabled = false;
    }

    public void SetOptions(PlayerOption[] options)
    {
        ClearButtons();
        _statusLabel.Visible = false;
        _statusLabel.RemoveThemeColorOverride("font_color");
        foreach (var opt in options)
        {
            var captured = opt;
            var b = new Button
            {
                Text = captured.Line,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                CustomMinimumSize = new Vector2(0, 40),
                Alignment = HorizontalAlignment.Left,
                AutowrapMode = TextServer.AutowrapMode.WordSmart,
                ClipText = false,
            };
            b.Pressed += () =>
            {
                if (!_enabled) return;
                EmitSignal(
                    SignalName.ActionSelected,
                    captured.ActionType,
                    captured.Detail,
                    captured.Line);
            };
            _buttonsBox.AddChild(b);
        }
        _enabled = true;
    }

    private void Build()
    {
        _root = new VBoxContainer
        {
            CustomMinimumSize = new Vector2(0, 110),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        AddChild(_root);

        _header = new Label { Text = "Your move:" };
        _root.AddChild(_header);

        _buttonsBox = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _root.AddChild(_buttonsBox);

        _statusLabel = new Label
        {
            Text = "",
            Visible = false,
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        _statusLabel.AddThemeColorOverride("font_color", new Color(0.7f, 0.7f, 0.7f));
        _root.AddChild(_statusLabel);
    }

    private void ClearButtons()
    {
        foreach (var child in _buttonsBox.GetChildren())
        {
            _buttonsBox.RemoveChild(child);
            child.QueueFree();
        }
    }

    private void ApplyEnabled(Node n)
    {
        foreach (var child in n.GetChildren())
        {
            if (child is Button b)
            {
                b.Disabled = !_enabled;
            }
            else
            {
                ApplyEnabled(child);
            }
        }
    }
}
