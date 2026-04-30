using Godot;

namespace Smallville;

/// <summary>
/// Right-column event log. New timeline events flash in as FLASH TRAFFIC
/// strips and persist as a chronological list of fired events with their
/// notification text. Visual style matches the dossier aesthetic — paper
/// cream rows with a kind-colored top rule.
/// </summary>
public partial class TimelineFeed : PanelContainer
{
    private static readonly Color PaperCream = new(0.94f, 0.90f, 0.82f);
    private static readonly Color InkBlack = new(0.10f, 0.10f, 0.11f);
    private static readonly Color InkMuted = new(0.30f, 0.28f, 0.26f);
    private static readonly Color ClassRed = new(0.61f, 0.17f, 0.17f);
    private static readonly Color SignalAmber = new(0.83f, 0.65f, 0.20f);

    private VBoxContainer _root = null!;
    private VBoxContainer _list = null!;

    public override void _Ready()
    {
        var bg = new StyleBoxFlat
        {
            BgColor = PaperCream,
            BorderColor = InkMuted,
            BorderWidthLeft = 1,
            BorderWidthRight = 1,
            BorderWidthTop = 1,
            BorderWidthBottom = 1,
            ContentMarginLeft = 8,
            ContentMarginRight = 8,
            ContentMarginTop = 8,
            ContentMarginBottom = 8,
        };
        AddThemeStyleboxOverride("panel", bg);

        _root = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        AddChild(_root);

        var header = new Label { Text = "FLASH TRAFFIC" };
        header.AddThemeFontSizeOverride("font_size", 11);
        header.AddThemeColorOverride("font_color", InkMuted);
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
        var topRule = new ColorRect
        {
            Color = KindColor(evt.Kind),
            CustomMinimumSize = new Vector2(0, 2),
        };
        var inner = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        inner.AddThemeConstantOverride("separation", 2);
        var col = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        col.AddThemeConstantOverride("separation", 0);
        col.AddChild(topRule);
        col.AddChild(inner);
        box.AddChild(col);

        var rowBg = new StyleBoxFlat
        {
            BgColor = new Color(1, 1, 1, 0),
            ContentMarginLeft = 6, ContentMarginRight = 6, ContentMarginTop = 4, ContentMarginBottom = 6,
        };
        box.AddThemeStyleboxOverride("panel", rowBg);

        var kind = new Label { Text = $"{evt.Kind.ToUpperInvariant()}  · @ ACTION {evt.AfterActions:D2}" };
        kind.AddThemeFontSizeOverride("font_size", 10);
        kind.AddThemeColorOverride("font_color", InkMuted);
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
        text.AddThemeFontSizeOverride("normal_font_size", 12);
        text.AddThemeColorOverride("default_color", InkBlack);
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
        note.AddThemeColorOverride("font_color", InkMuted);
        note.AddThemeFontSizeOverride("font_size", 10);
        _list.AddChild(note);
        _list.MoveChild(note, 0);
    }

    private static Color KindColor(string kind) => kind switch
    {
        "world_event" => SignalAmber,
        "flash_traffic" => ClassRed,
        "directive" => InkBlack,
        _ => InkMuted,
    };

    private void Flash(Control target)
    {
        var tween = target.CreateTween();
        target.Modulate = new Color(1f, 1f, 0.65f);
        tween.TweenProperty(target, "modulate", Colors.White, 0.9f);
    }
}
