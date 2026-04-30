using System.Collections.Generic;
using Godot;

namespace Smallville;

/// <summary>
/// Left-edge dossier tabs: one per agent. Clicking a tab emits
/// <see cref="ChannelSelected"/>. Inactive tabs show an unread mark when their
/// channel has scene-update injections the player hasn't viewed since the last
/// switch. Lost agents render as silent (greyed, no click).
///
/// Visual aim: filing-cabinet tabs. Active tab is wider with an amber accent
/// strip on its left edge and shows the agent's full callsign, city, and
/// last-contact line. Inactive tabs are narrower and show callsign + city only.
/// </summary>
public partial class DossierTabs : VBoxContainer
{
    [Signal] public delegate void ChannelSelectedEventHandler(string agentId);

    private static readonly Color InkBlack = new(0.10f, 0.10f, 0.11f);
    private static readonly Color PaperCream = new(0.94f, 0.90f, 0.82f);
    private static readonly Color PaperCreamMuted = new(0.86f, 0.83f, 0.76f);
    private static readonly Color PaperCreamDim = new(0.78f, 0.75f, 0.69f);
    private static readonly Color SignalAmber = new(0.83f, 0.65f, 0.20f);
    private static readonly Color ClassRed = new(0.61f, 0.17f, 0.17f);
    private static readonly Color InkMuted = new(0.30f, 0.28f, 0.26f);

    private readonly Dictionary<string, TabRow> _rows = new();
    private string _activeAgentId = "";
    private readonly HashSet<string> _unread = new();

    public override void _Ready()
    {
        AddThemeConstantOverride("separation", 4);
        SizeFlagsHorizontal = SizeFlags.ShrinkBegin;
    }

    public void SetAgents(IList<AgentDef> agents)
    {
        ClearRows();
        foreach (var a in agents)
        {
            var row = new TabRow(a, OnTabPressed);
            AddChild(row);
            _rows[a.Id] = row;
        }
        if (agents.Count > 0)
        {
            SetActive(agents[0].Id);
        }
    }

    public void SetActive(string agentId)
    {
        _activeAgentId = agentId;
        _unread.Remove(agentId);
        foreach (var (id, row) in _rows)
        {
            row.SetActive(id == agentId);
            row.SetUnread(_unread.Contains(id));
        }
    }

    public string ActiveAgentId => _activeAgentId;

    public void MarkUnread(string agentId)
    {
        if (agentId == _activeAgentId) return;          // active channel is read-by-construction
        if (!_rows.ContainsKey(agentId)) return;
        _unread.Add(agentId);
        _rows[agentId].SetUnread(true);
    }

    public void UpdateFrom(HandlerState state)
    {
        foreach (var a in state.Agents)
        {
            if (!_rows.TryGetValue(a.Id, out var row)) continue;
            row.SetLost(a.Lost);
            row.SetLastContact(a.LastReportActionCount);
        }
    }

    private void OnTabPressed(string agentId)
    {
        if (_rows.TryGetValue(agentId, out var row) && row.IsLost) return;
        if (agentId == _activeAgentId) return;
        SetActive(agentId);
        EmitSignal(SignalName.ChannelSelected, agentId);
    }

    private void ClearRows()
    {
        foreach (var (_, row) in _rows) row.QueueFree();
        _rows.Clear();
        _unread.Clear();
        _activeAgentId = "";
    }

    // ───────────────────────── single tab ─────────────────────────

    private partial class TabRow : Button
    {
        private readonly AgentDef _def;
        private readonly System.Action<string> _onPressed;
        private Label _callsign = null!;
        private Label _city = null!;
        private Label _contact = null!;
        private Label _newBadge = null!;
        private ColorRect _activeAccent = null!;
        private bool _active;
        private int _lastContactCount = -1;

        public bool IsLost { get; private set; }

        public TabRow(AgentDef def, System.Action<string> onPressed)
        {
            _def = def;
            _onPressed = onPressed;
            Build();
            Pressed += () => _onPressed(_def.Id);
        }

        public void SetActive(bool active)
        {
            if (_active == active) return;
            _active = active;
            CustomMinimumSize = new Vector2(active ? 168 : 124, 64);
            _activeAccent.Visible = active;
            _contact.Visible = active;
            ApplyStyle();
        }

        public void SetUnread(bool unread) => _newBadge.Visible = unread && !_active;

        public void SetLost(bool lost)
        {
            if (IsLost == lost) return;
            IsLost = lost;
            Disabled = lost;
            ApplyStyle();
        }

        public void SetLastContact(int actionCount)
        {
            if (actionCount == _lastContactCount) return;
            _lastContactCount = actionCount;
            _contact.Text = actionCount > 0 ? $"last @ {actionCount:D2}" : "no contact";
        }

        private void Build()
        {
            CustomMinimumSize = new Vector2(124, 64);
            Flat = true;
            FocusMode = FocusModeEnum.None;
            ToggleMode = false;
            ClipText = true;

            // Button doesn't lay out children — we overlay a sibling Control with MouseFilter.Ignore so clicks fall through.
            var overlay = new Control
            {
                MouseFilter = MouseFilterEnum.Ignore,
                AnchorRight = 1, AnchorBottom = 1,
                OffsetLeft = 8, OffsetRight = -6, OffsetTop = 6, OffsetBottom = -6,
            };
            AddChild(overlay);

            _activeAccent = new ColorRect
            {
                Color = SignalAmber,
                MouseFilter = MouseFilterEnum.Ignore,
                AnchorTop = 0, AnchorBottom = 1,
                OffsetLeft = 0, OffsetRight = 3, OffsetTop = 0, OffsetBottom = 0,
                Visible = false,
            };
            var accentHost = new Control { MouseFilter = MouseFilterEnum.Ignore };
            accentHost.SetAnchorsAndOffsetsPreset(LayoutPreset.LeftWide);
            accentHost.OffsetRight = 4;
            AddChild(accentHost);
            accentHost.AddChild(_activeAccent);
            _activeAccent.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);

            _callsign = new Label
            {
                Text = _def.Callsign,
                AnchorTop = 0, OffsetTop = 2, OffsetRight = -8,
            };
            _callsign.AddThemeFontSizeOverride("font_size", 17);
            _callsign.AddThemeColorOverride("font_color", InkBlack);
            HandlerFonts.ApplyStencil(_callsign);
            overlay.AddChild(_callsign);

            _city = new Label
            {
                Text = _def.City,
                AnchorTop = 0, OffsetTop = 24, OffsetRight = -8,
            };
            _city.AddThemeFontSizeOverride("font_size", 11);
            _city.AddThemeColorOverride("font_color", InkMuted);
            overlay.AddChild(_city);

            _contact = new Label
            {
                Text = "no contact",
                AnchorTop = 0, OffsetTop = 42, OffsetRight = -8,
                Visible = false,
            };
            _contact.AddThemeFontSizeOverride("font_size", 10);
            _contact.AddThemeColorOverride("font_color", InkMuted);
            overlay.AddChild(_contact);

            _newBadge = new Label
            {
                Text = "● NEW",
                AnchorRight = 1, AnchorTop = 0,
                OffsetLeft = -50, OffsetTop = 4, OffsetRight = -4,
                Visible = false,
                HorizontalAlignment = HorizontalAlignment.Right,
            };
            _newBadge.AddThemeFontSizeOverride("font_size", 10);
            _newBadge.AddThemeColorOverride("font_color", ClassRed);
            HandlerFonts.ApplyStencil(_newBadge);
            overlay.AddChild(_newBadge);

            ApplyStyle();
        }

        private void ApplyStyle()
        {
            var fill = _active ? PaperCream : PaperCreamMuted;
            if (IsLost) fill = PaperCreamDim;

            var box = new StyleBoxFlat
            {
                BgColor = fill,
                BorderColor = InkMuted,
                BorderWidthTop = 0,
                BorderWidthBottom = 1,
                BorderWidthLeft = 0,
                BorderWidthRight = 0,
                ContentMarginLeft = 8,
                ContentMarginRight = 6,
                ContentMarginTop = 6,
                ContentMarginBottom = 6,
            };
            AddThemeStyleboxOverride("normal", box);
            AddThemeStyleboxOverride("hover", box);
            AddThemeStyleboxOverride("pressed", box);
            AddThemeStyleboxOverride("focus", box);
            AddThemeStyleboxOverride("disabled", box);

            var textColor = IsLost ? new Color(InkMuted, 0.5f) : InkBlack;
            _callsign.AddThemeColorOverride("font_color", textColor);
        }
    }
}
