using System.Collections.Generic;
using Godot;

namespace Smallville;

/// <summary>
/// Right-side panel showing each hostage with a color-coded status badge.
/// </summary>
public partial class HostageStatusPanel : PanelContainer
{
    private VBoxContainer _root = null!;
    private readonly Dictionary<string, Label> _statusLabels = new();
    private readonly Dictionary<string, string> _lastStatus = new();

    public override void _Ready()
    {
        _root = new VBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        AddChild(_root);

        var header = new Label { Text = "Hostages" };
        header.AddThemeFontSizeOverride("font_size", 18);
        _root.AddChild(header);
        _root.AddChild(new HSeparator());
    }

    public void SetHostages(List<Hostage> hostages)
    {
        // Clear any existing rows beyond the header + separator
        var children = _root.GetChildren();
        for (int i = children.Count - 1; i >= 2; i--)
        {
            _root.RemoveChild(children[i]);
            children[i].QueueFree();
        }
        _statusLabels.Clear();
        _lastStatus.Clear();

        foreach (var h in hostages)
        {
            var box = new VBoxContainer
            {
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
            };
            box.AddThemeConstantOverride("separation", 2);

            var name = new Label { Text = h.Name };
            name.AddThemeFontSizeOverride("font_size", 14);
            box.AddChild(name);

            var desc = new Label
            {
                Text = h.Description,
                AutowrapMode = TextServer.AutowrapMode.WordSmart,
            };
            desc.AddThemeFontSizeOverride("font_size", 11);
            desc.AddThemeColorOverride("font_color", new Color(0.7f, 0.7f, 0.7f));
            box.AddChild(desc);

            var status = new Label { Text = StatusBadge(h.Status) };
            status.AddThemeColorOverride("font_color", StatusColor(h.Status));
            box.AddChild(status);

            _statusLabels[h.Name] = status;
            _lastStatus[h.Name] = h.Status;
            _root.AddChild(box);
            _root.AddChild(new HSeparator());
        }
    }

    public void UpdateFrom(List<Hostage> hostages)
    {
        foreach (var h in hostages)
        {
            if (_lastStatus.TryGetValue(h.Name, out var prev) && prev == h.Status) continue;
            if (!_statusLabels.TryGetValue(h.Name, out var label)) continue;
            label.Text = StatusBadge(h.Status);
            label.AddThemeColorOverride("font_color", StatusColor(h.Status));
            _lastStatus[h.Name] = h.Status;
        }
    }

    private static string StatusBadge(string status) => status switch
    {
        "held" => "● HELD",
        "released" => "✓ RELEASED",
        "harmed" => "✗ HARMED",
        _ => status,
    };

    private static Color StatusColor(string status) => status switch
    {
        "held" => new Color(0.95f, 0.7f, 0.2f),       // amber
        "released" => new Color(0.4f, 0.85f, 0.4f),   // green
        "harmed" => new Color(0.9f, 0.3f, 0.3f),      // red
        _ => Colors.White,
    };
}
