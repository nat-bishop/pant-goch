using Godot;

namespace Smallville;

/// <summary>
/// Scrolling notification feed for timeline events. New events flash briefly.
/// </summary>
public partial class TimelineFeed : PanelContainer
{
    private VBoxContainer _root = null!;
    private VBoxContainer _list = null!;

    public override void _Ready()
    {
        _root = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        AddChild(_root);

        var header = new Label { Text = "Comms" };
        header.AddThemeFontSizeOverride("font_size", 16);
        _root.AddChild(header);
        _root.AddChild(new HSeparator());

        var scroll = new ScrollContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0, 220),
        };
        _list = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _list.AddThemeConstantOverride("separation", 6);
        scroll.AddChild(_list);
        _root.AddChild(scroll);
    }

    public void AddEvent(TimelineEvent evt)
    {
        var box = new PanelContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        var inner = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        box.AddChild(inner);

        var kind = new Label { Text = $"[{evt.Kind.ToUpper()}]  @ action {evt.AfterActions}" };
        kind.AddThemeFontSizeOverride("font_size", 11);
        kind.AddThemeColorOverride("font_color", KindColor(evt.Kind));
        inner.AddChild(kind);

        var text = new RichTextLabel
        {
            BbcodeEnabled = false,
            FitContent = true,
            ScrollActive = false,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            Text = evt.NotificationText,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        text.AddThemeFontSizeOverride("normal_font_size", 13);
        inner.AddChild(text);

        // newest at top
        _list.AddChild(box);
        _list.MoveChild(box, 0);

        Flash(box);
    }

    public void AddSystemNote(string text)
    {
        var note = new Label
        {
            Text = $"— {text} —",
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        note.AddThemeColorOverride("font_color", new Color(0.6f, 0.6f, 0.6f));
        note.AddThemeFontSizeOverride("font_size", 11);
        _list.AddChild(note);
        _list.MoveChild(note, 0);
    }

    private static Color KindColor(string kind) => kind switch
    {
        "check_in" => new Color(0.6f, 0.7f, 0.95f),
        "medical" => new Color(0.95f, 0.7f, 0.5f),
        "ultimatum" => new Color(0.95f, 0.4f, 0.4f),
        _ => Colors.White,
    };

    private void Flash(Control target)
    {
        var tween = target.CreateTween();
        target.Modulate = new Color(1, 1, 0.6f);
        tween.TweenProperty(target, "modulate", Colors.White, 1.0f);
    }
}
