using System.Collections.Generic;
using Godot;

namespace Smallville;

/// <summary>
/// "DECISION" row at the bottom of the screen — 2-5 raised paper cards,
/// each carrying a tag + a hidden_line revealed on hover. Selecting a card
/// emits <see cref="OptionSelected"/> with the hidden_line, and triggers a
/// short *CONFIRMED* stamp animation so committing an action *feels* like
/// committing.
///
/// During streaming, all cards display a "RECEIVING TRANSMISSION" cross-stamp
/// — physically blocked, not just disabled-grey.
/// </summary>
public partial class ActionPanel : PanelContainer
{
    [Signal] public delegate void OptionSelectedEventHandler(string hiddenLine);

    private static readonly Color PaperCream = new(0.94f, 0.90f, 0.82f);
    private static readonly Color PaperCreamMuted = new(0.86f, 0.83f, 0.76f);
    private static readonly Color InkBlack = new(0.10f, 0.10f, 0.11f);
    private static readonly Color InkMuted = new(0.30f, 0.28f, 0.26f);
    private static readonly Color SignalAmber = new(0.83f, 0.65f, 0.20f);
    private static readonly Color ClassRed = new(0.61f, 0.17f, 0.17f);

    private VBoxContainer _root = null!;
    private Label _header = null!;
    private RichTextLabel _previewLine = null!;
    private HBoxContainer _cardRow = null!;
    private Label _statusLabel = null!;
    private Label _streamStamp = null!;
    private Label _confirmStamp = null!;

    private readonly List<DecisionCard> _cards = new();
    private bool _enabled;

    public override void _Ready()
    {
        Build();
    }

    public void SetOptions(DialogueOption[] options)
    {
        ClearCards();
        foreach (var opt in options)
        {
            var card = new DecisionCard(opt);
            card.HoverPreview += text => SetPreview(text);
            card.HoverEnded += () => SetPreview("");
            card.Selected += hidden =>
            {
                if (!_enabled) return;
                EmitSignal(SignalName.OptionSelected, hidden);
                PlayConfirmStamp();
            };
            _cardRow.AddChild(card);
            _cards.Add(card);
        }
        _statusLabel.Visible = false;
    }

    public void SetEnabled(bool enabled)
    {
        _enabled = enabled;
        foreach (var c in _cards) c.SetInteractive(enabled);
        _streamStamp.Visible = !enabled && _cards.Count > 0;
    }

    public void GrabInputFocus()
    {
        if (_cards.Count > 0) _cards[0].GrabFocus();
    }

    public void ShowError(string message)
    {
        ClearCards();
        _statusLabel.Text = $"// SYSTEM: {message}";
        _statusLabel.AddThemeColorOverride("font_color", ClassRed);
        _statusLabel.Visible = true;
    }

    public void ShowLoading(string text = "AWAITING SIGNAL…")
    {
        ClearCards();
        _statusLabel.Text = $"// {text}";
        _statusLabel.AddThemeColorOverride("font_color", InkMuted);
        _statusLabel.Visible = true;
    }

    private void SetPreview(string text)
    {
        _previewLine.Text = string.IsNullOrEmpty(text) ? "" : $"[i]{Escape(text)}[/i]";
    }

    private static string Escape(string s) => s.Replace("[", "[lb]");

    private void PlayConfirmStamp()
    {
        _confirmStamp.Visible = true;
        _confirmStamp.Modulate = new Color(ClassRed, 0);
        _confirmStamp.Scale = new Vector2(1.6f, 1.6f);
        var tween = CreateTween();
        tween.SetParallel(true);
        tween.TweenProperty(_confirmStamp, "modulate", new Color(ClassRed, 1), 0.08);
        tween.TweenProperty(_confirmStamp, "scale", Vector2.One, 0.18).SetTrans(Tween.TransitionType.Cubic).SetEase(Tween.EaseType.Out);
        tween.Chain().TweenProperty(_confirmStamp, "modulate", new Color(ClassRed, 0), 0.18);
        tween.TweenCallback(Callable.From(() => _confirmStamp.Visible = false));
    }

    private void ClearCards()
    {
        foreach (var c in _cards) c.QueueFree();
        _cards.Clear();
        SetPreview("");
    }

    private void Build()
    {
        // Outer panel: paper background with subtle dropshadow effect via thicker top border.
        var bg = new StyleBoxFlat
        {
            BgColor = PaperCream,
            BorderColor = InkMuted,
            BorderWidthTop = 2,
            BorderWidthBottom = 0,
            BorderWidthLeft = 0,
            BorderWidthRight = 0,
            ContentMarginLeft = 16,
            ContentMarginRight = 16,
            ContentMarginTop = 10,
            ContentMarginBottom = 12,
            ShadowColor = new Color(0, 0, 0, 0.20f),
            ShadowSize = 6,
            ShadowOffset = new Vector2(0, -3),
        };
        AddThemeStyleboxOverride("panel", bg);

        _root = new VBoxContainer
        {
            CustomMinimumSize = new Vector2(0, 100),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _root.AddThemeConstantOverride("separation", 4);
        AddChild(_root);

        // Header line: "DECISION"
        var headerRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _root.AddChild(headerRow);
        _header = new Label { Text = "DECISION" };
        _header.AddThemeFontSizeOverride("font_size", 11);
        _header.AddThemeColorOverride("font_color", InkMuted);
        headerRow.AddChild(_header);
        // hairline rule to the right of the header
        var rule = new ColorRect
        {
            Color = new Color(InkMuted, 0.35f),
            CustomMinimumSize = new Vector2(0, 1),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ShrinkCenter,
        };
        var ruleHost = new Control
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ShrinkCenter,
            CustomMinimumSize = new Vector2(0, 12),
        };
        ruleHost.AddChild(rule);
        rule.SetAnchorsAndOffsetsPreset(LayoutPreset.CenterLeft);
        rule.OffsetLeft = 10;
        rule.OffsetRight = 0;
        rule.AnchorRight = 1;
        headerRow.AddChild(ruleHost);

        // Hover preview line — italicized hidden_line for the focused card.
        _previewLine = new RichTextLabel
        {
            BbcodeEnabled = true,
            FitContent = true,
            ScrollActive = false,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            CustomMinimumSize = new Vector2(0, 22),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _previewLine.AddThemeFontSizeOverride("normal_font_size", 12);
        _previewLine.AddThemeColorOverride("default_color", InkMuted);
        _root.AddChild(_previewLine);

        // Card row.
        _cardRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _cardRow.AddThemeConstantOverride("separation", 10);
        _root.AddChild(_cardRow);

        // Status label (loading / error).
        _statusLabel = new Label
        {
            Text = "",
            Visible = false,
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        _statusLabel.AddThemeFontSizeOverride("font_size", 12);
        _root.AddChild(_statusLabel);

        // Stream cross-stamp.
        _streamStamp = new Label
        {
            Text = "[ RECEIVING TRANSMISSION ]",
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            Visible = false,
            ZIndex = 10,
            MouseFilter = MouseFilterEnum.Ignore,
        };
        _streamStamp.AddThemeFontSizeOverride("font_size", 16);
        _streamStamp.AddThemeColorOverride("font_color", ClassRed);
        HandlerFonts.ApplyStencil(_streamStamp);
        _streamStamp.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        AddChild(_streamStamp);

        // Confirm stamp overlay (one-shot animated).
        _confirmStamp = new Label
        {
            Text = "*CONFIRMED*",
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            Visible = false,
            ZIndex = 11,
            MouseFilter = MouseFilterEnum.Ignore,
        };
        _confirmStamp.AddThemeFontSizeOverride("font_size", 26);
        _confirmStamp.AddThemeColorOverride("font_color", ClassRed);
        HandlerFonts.ApplyStencil(_confirmStamp);
        _confirmStamp.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        AddChild(_confirmStamp);
    }

    // ───────────────────────── card ─────────────────────────

    private partial class DecisionCard : PanelContainer
    {
        [Signal] public delegate void HoverPreviewEventHandler(string text);
        [Signal] public delegate void HoverEndedEventHandler();
        [Signal] public delegate void SelectedEventHandler(string hiddenLine);

        private readonly DialogueOption _opt;
        private Label _tagLabel = null!;
        private bool _interactive;

        public DecisionCard(DialogueOption opt)
        {
            _opt = opt;
            Build();
        }

        public void SetInteractive(bool on)
        {
            _interactive = on;
            Modulate = new Color(1, 1, 1, on ? 1f : 0.5f);
            MouseFilter = on ? MouseFilterEnum.Stop : MouseFilterEnum.Ignore;
        }

        public new void GrabFocus() => GrabClickFocus();

        private void Build()
        {
            CustomMinimumSize = new Vector2(140, 56);
            SizeFlagsHorizontal = SizeFlags.ExpandFill;
            SizeFlagsVertical = SizeFlags.ShrinkCenter;
            ApplyStyle(hover: false);

            var col = new VBoxContainer
            {
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                SizeFlagsVertical = SizeFlags.ExpandFill,
            };
            AddChild(col);

            _tagLabel = new Label
            {
                Text = _opt.Tag.ToUpperInvariant(),
                HorizontalAlignment = HorizontalAlignment.Center,
            };
            _tagLabel.AddThemeFontSizeOverride("font_size", 15);
            _tagLabel.AddThemeColorOverride("font_color", InkBlack);
            HandlerFonts.ApplyStencil(_tagLabel);
            col.AddChild(_tagLabel);

            MouseEntered += () =>
            {
                if (!_interactive) return;
                ApplyStyle(hover: true);
                EmitSignal(SignalName.HoverPreview, _opt.HiddenLine);
            };
            MouseExited += () =>
            {
                ApplyStyle(hover: false);
                EmitSignal(SignalName.HoverEnded);
            };
            GuiInput += @event =>
            {
                if (!_interactive) return;
                if (@event is InputEventMouseButton mb && mb.Pressed && mb.ButtonIndex == MouseButton.Left)
                {
                    EmitSignal(SignalName.Selected, _opt.HiddenLine);
                    GetViewport().SetInputAsHandled();
                }
            };
        }

        private void ApplyStyle(bool hover)
        {
            var fill = hover ? PaperCream : PaperCreamMuted;
            var box = new StyleBoxFlat
            {
                BgColor = fill,
                BorderColor = hover ? SignalAmber : InkMuted,
                BorderWidthTop = 1,
                BorderWidthBottom = 1,
                BorderWidthLeft = 1,
                BorderWidthRight = 1,
                ContentMarginLeft = 12,
                ContentMarginRight = 12,
                ContentMarginTop = 12,
                ContentMarginBottom = 12,
                ShadowColor = new Color(0, 0, 0, hover ? 0.18f : 0.10f),
                ShadowSize = hover ? 5 : 2,
                ShadowOffset = new Vector2(0, hover ? 3 : 2),
            };
            AddThemeStyleboxOverride("panel", box);
        }
    }
}
