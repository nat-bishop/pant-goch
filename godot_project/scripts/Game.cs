using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Godot;

namespace Smallville;

/// <summary>Scene controller. See CLAUDE.md for architecture and invariants.</summary>
public partial class Game : Control
{
    private sealed class ChannelView
    {
        public AgentDef Def { get; }
        public StringBuilder Cache { get; } = new();
        public int TxCounter;
        public ChannelView(AgentDef def) { Def = def; }
    }

    // ── palette ──
    private static readonly Color PaperCream = new(0.94f, 0.90f, 0.82f);
    private static readonly Color InkBlack = new(0.10f, 0.10f, 0.11f);
    private static readonly Color InkMuted = new(0.30f, 0.28f, 0.26f);
    private static readonly Color SignalAmber = new(0.83f, 0.65f, 0.20f);
    private static readonly Color ClassRed = new(0.61f, 0.17f, 0.17f);
    private static readonly Color DeskGrey = new(0.18f, 0.17f, 0.16f);

    // ── nodes ──
    private LLMClient _client = null!;
    private Label _budgetCount = null!;
    private BudgetPunchcard _punchcard = null!;
    private Label _opLabel = null!;
    private DossierTabs _tabs = null!;
    private RichTextLabel _chatLog = null!;
    private ActionPanel _actionPanel = null!;
    private Button _stopButton = null!;
    private TimelineFeed _timelineFeed = null!;
    private DebugPanel? _debugPanel;
    private VBoxContainer _rightColumn = null!;
    private Control _bootSplash = null!;
    private Label _bootMessage = null!;
    private Control _resolutionOverlay = null!;
    private Label _resolutionTitle = null!;
    private RichTextLabel _resolutionBody = null!;

    // ── state ──
    private Scenario _scenario = new();
    private HandlerState _state = new();
    private CancellationTokenSource? _activeStreamCts;
    private CancellationTokenSource? _optionsCts;
    private CancellationTokenSource? _eventsCts;
    private bool _streaming;
    private bool _ended;
    private bool _budgetAdvisoryStamped;

    private readonly Dictionary<string, ChannelView> _channels = new();

    // ── lifecycle ──

    public override void _Ready()
    {
        _client = GetNode<LLMClient>("/root/LLMClient");
        BuildUI();
        ShowBoot("Booting…");
        _ = BootAsync();
    }

    public override void _ExitTree()
    {
        _activeStreamCts?.Cancel();
        _optionsCts?.Cancel();
        _eventsCts?.Cancel();
    }

    // ── boot ──

    private async Task BootAsync()
    {
        try
        {
            ShowBoot("Connecting to station…");
            await _client.WaitReady(180.0);

            ShowBoot("Loading operation brief…");
            var scenarioTask = _client.GetScenarioAsync();
            var stateTask = _client.GetStateAsync();
            _scenario = await scenarioTask;
            _state = await stateTask;

            _opLabel.Text = $"{_scenario.Operation.Codename} — {_scenario.Operation.Objective}";

            var firstId = SeedChannelsAndRenderFirst();
            UpdateBudgetUI();
            MaybeMountDebugPanel();
            _debugPanel?.SetActiveAgent(firstId);

            _eventsCts = new CancellationTokenSource();
            _ = _client.SubscribeEventsAsync(
                onGameEnded: er => OnGameEnded(er),
                onError: msg => GD.PrintErr($"[events] {msg}"),
                ct: _eventsCts.Token);

            HideBoot();

            _actionPanel.ShowLoading();
            await RefreshOptionsAsync(firstId);
        }
        catch (Exception e)
        {
            GD.PrintErr($"[Game] boot failed: {e}");
            ShowBoot($"Boot failed: {e.Message}\n\nIs the sidecar running? Is llama-server up?");
        }
    }

    // ── action flow ──

    private void OnOptionSelected(string hiddenLine)
    {
        if (_ended) return;
        var aid = _tabs.ActiveAgentId;
        if (string.IsNullOrEmpty(aid)) return;
        _ = FireAction(aid, hiddenLine);
    }

    private async Task FireAction(string agentId, string line)
    {
        _activeStreamCts?.Cancel();
        _activeStreamCts = new CancellationTokenSource();
        var ct = _activeStreamCts.Token;

        SetStreamingUI(true);
        AppendHandlerLine(agentId, line);
        BeginAgentBubble(agentId);

        try
        {
            await _client.StreamActionAsync(
                agentId, line,
                onTextDelta: (id, delta) => OnAgentDelta(id, delta),
                onState: ApplyStateEvent,
                onTimeline: OnTimelineEvent,
                onDone: OnStreamDone,
                onError: OnStreamError,
                ct: ct);
        }
        catch (Exception e)
        {
            GD.PrintErr($"[Game] action stream failed: {e}");
            OnStreamError(e.Message);
        }
    }

    private void OnAgentDelta(string agentId, string delta)
    {
        var bb = EscapeBb(delta);
        AppendToChannelCache(agentId, bb);
        if (agentId == _tabs.ActiveAgentId) _chatLog.AppendText(bb);
    }

    private void OnStopPressed()
    {
        _activeStreamCts?.Cancel();
        MarkInterrupted(_tabs.ActiveAgentId);
        RecoverFromStream();
    }

    private void OnStreamDone() => RecoverFromStream();

    private void OnStreamError(string msg)
    {
        AppendSystemToChannel(_tabs.ActiveAgentId, $"// SIGNAL ERROR: {msg}");
        RecoverFromStream();
    }

    private void RecoverFromStream()
    {
        SetStreamingUI(false);
        if (_ended) return;
        _actionPanel.ShowLoading();
        _ = RefreshOptionsAsync(_tabs.ActiveAgentId);
    }

    private async Task RefreshOptionsAsync(string agentId)
    {
        if (string.IsNullOrEmpty(agentId)) return;
        _optionsCts?.Cancel();
        _optionsCts = new CancellationTokenSource();
        var ct = _optionsCts.Token;
        try
        {
            var options = await _client.GetOptionsAsync(agentId, ct);
            if (ct.IsCancellationRequested || _ended) return;
            if (agentId != _tabs.ActiveAgentId) return;  // user switched mid-fetch
            if (options.Length == 0)
            {
                _actionPanel.ShowError("no options available.");
                return;
            }
            _actionPanel.SetOptions(options);
            _actionPanel.SetEnabled(true);
            _actionPanel.GrabInputFocus();
        }
        catch (OperationCanceledException) { /* expected */ }
        catch (Exception e)
        {
            GD.PrintErr($"[Game] /options failed: {e}");
            _actionPanel.ShowError($"options unavailable ({e.Message}).");
        }
    }

    private void ApplyStateEvent(HandlerState s)
    {
        _state = s;
        _tabs.UpdateFrom(s);
        UpdateBudgetUI();
        if (s.Debug != null)
        {
            MaybeMountDebugPanel();
            _debugPanel?.UpdateFrom(s.Debug);
        }
        if (s.Ended)
        {
            EndGame(s.EndReason ?? "ended");
        }
    }

    private void OnTimelineEvent(TimelineEvent evt)
    {
        _timelineFeed.AddEvent(evt);
        var active = _tabs.ActiveAgentId;
        foreach (var aid in evt.AffectedAgents)
        {
            AppendSystemToChannel(aid, $"[scene update] {evt.NotificationText}");
            if (aid != active) _tabs.MarkUnread(aid);
        }
    }

    private void OnGameEnded(string reason)
    {
        EndGame(reason);
    }

    // ── channel switching ──

    private void OnChannelSelected(string agentId)
    {
        if (_streaming)
        {
            _activeStreamCts?.Cancel();
        }
        RenderChannel(agentId);
        _debugPanel?.SetActiveAgent(agentId);
        if (!_ended)
        {
            _actionPanel.ShowLoading();
            _ = RefreshOptionsAsync(agentId);
        }
    }

    private void RenderChannel(string agentId)
    {
        if (string.IsNullOrEmpty(agentId)) return;
        _chatLog.Clear();
        if (_channels.TryGetValue(agentId, out var ch))
        {
            _chatLog.AppendText(ch.Cache.ToString());
        }
    }

    // ── chat cache helpers ──

    private string SeedChannelsAndRenderFirst()
    {
        _channels.Clear();
        _chatLog.Clear();
        // Seed BEFORE _tabs.SetAgents — until then ActiveAgentId is "" so the
        // Append* helpers skip their live-render branch and only fill the cache.
        foreach (var a in _scenario.Agents) _channels[a.Id] = new ChannelView(a);
        var opLine = $"OPERATION {_scenario.Operation.Codename} :: {_scenario.Operation.Objective}";
        foreach (var a in _scenario.Agents)
        {
            AppendSystemToChannel(a.Id, _scenario.Setup);
            AppendSystemToChannel(a.Id, opLine);
            BeginAgentBubble(a.Id);
            AppendToChannelCache(a.Id, EscapeBb(a.OpeningTransmission));
        }
        _tabs.SetAgents(_scenario.Agents);
        _tabs.UpdateFrom(_state);
        var firstId = _scenario.Agents.Count > 0 ? _scenario.Agents[0].Id : "";
        RenderChannel(firstId);
        return firstId;
    }

    private void AppendToChannelCache(string agentId, string bb)
    {
        if (!_channels.TryGetValue(agentId, out var ch)) return;
        ch.Cache.Append(bb);
    }

    private void AppendHandlerLine(string agentId, string line)
    {
        var bb = $"\n\n[color=#d4a533][b]HANDLER:[/b][/color] {EscapeBb(line)}";
        AppendToChannelCache(agentId, bb);
        if (agentId == _tabs.ActiveAgentId) _chatLog.AppendText(bb);
    }

    private void BeginAgentBubble(string agentId)
    {
        if (!_channels.TryGetValue(agentId, out var ch)) return;
        ch.TxCounter += 1;
        var time = DateTime.UtcNow.ToString("HHmm", CultureInfo.InvariantCulture) + "Z";
        var header = $"\n\n[color=#1a1a1c][b]{ch.Def.Callsign}[/b][/color] / {ch.Def.City} / {time} / TX-{ch.TxCounter:D3}\n";
        ch.Cache.Append(header);
        if (agentId == _tabs.ActiveAgentId) _chatLog.AppendText(header);
    }

    private void AppendSystemToChannel(string agentId, string text)
    {
        var bb = $"\n\n[color=#4d4944][i]{EscapeBb(text)}[/i][/color]";
        AppendToChannelCache(agentId, bb);
        if (agentId == _tabs.ActiveAgentId) _chatLog.AppendText(bb);
    }

    private void MarkInterrupted(string agentId)
    {
        const string bb = " [color=#a07a7a][i] [interrupted][/i][/color]";
        AppendToChannelCache(agentId, bb);
        if (agentId == _tabs.ActiveAgentId) _chatLog.AppendText(bb);
    }

    private static string EscapeBb(string s) =>
        s.IndexOf('[') < 0 ? s : s.Replace("[", "[lb]");

    // ── ui state ──

    private void SetStreamingUI(bool streaming)
    {
        _streaming = streaming;
        _actionPanel.SetEnabled(!streaming);
        _stopButton.Visible = streaming;
    }

    private void UpdateBudgetUI()
    {
        var remaining = _state.ActionsRemaining;
        var max = _state.MaxActions > 0 ? _state.MaxActions : _scenario.MaxActions;
        var used = max - remaining;
        _punchcard.Set(used, max);
        _budgetCount.Text = $"{remaining:D2}/{max:D2} DECISIONS";
        if (!_budgetAdvisoryStamped && remaining > 0 && remaining <= 5)
        {
            _budgetAdvisoryStamped = true;
            var aid = _tabs.ActiveAgentId;
            if (!string.IsNullOrEmpty(aid))
            {
                AppendSystemToChannel(aid, "BUDGET ADVISORY: five decisions remaining before window closes.");
            }
        }
    }

    private void MaybeMountDebugPanel()
    {
        if (_debugPanel != null) return;
        if (_state.Debug == null) return;
        _debugPanel = new DebugPanel { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _rightColumn.AddChild(_debugPanel);
        _debugPanel.UpdateFrom(_state.Debug);
        _debugPanel.SetActiveAgent(_tabs.ActiveAgentId);
    }

    private void EndGame(string reason)
    {
        if (_ended) return;
        _ended = true;
        _activeStreamCts?.Cancel();
        _optionsCts?.Cancel();
        SetStreamingUI(false);
        _actionPanel.SetEnabled(false);

        _resolutionTitle.Text = OutcomeTitle(reason);
        _resolutionBody.Clear();
        _resolutionBody.AppendText($"\n\nFile: [b]{_scenario.Operation.Codename}[/b]");
        _resolutionBody.AppendText($"\nReason: [b]{ReasonLabel(reason)}[/b]");
        var max = _state.MaxActions > 0 ? _state.MaxActions : _scenario.MaxActions;
        var used = max - _state.ActionsRemaining;
        _resolutionBody.AppendText($"\nDecisions used: {used:D2} of {max:D2}");
        _resolutionBody.AppendText("\n\n[b]ASSET REGISTRY[/b]");
        foreach (var a in _state.Agents)
        {
            var name = _channels.TryGetValue(a.Id, out var ch) ? ch.Def.Callsign : a.Callsign;
            var city = _channels.TryGetValue(a.Id, out var ch2) ? ch2.Def.City : a.City;
            var status = a.Lost ? "[color=#9b2c2c]LOST[/color]" : "[color=#1a1a1c]ACTIVE[/color]";
            var lastContact = a.LastReportActionCount > 0
                ? $"last @ action {a.LastReportActionCount:D2}"
                : "no contact";
            _resolutionBody.AppendText($"\n  · {name} / {city} — {status} ({lastContact})");
        }
        _resolutionOverlay.Visible = true;
    }

    private static string ReasonLabel(string reason) => reason switch
    {
        "budget_exhausted" => "decision window closed",
        "all_lost" => "all assets lost",
        _ => reason,
    };

    private static string OutcomeTitle(string reason) => reason switch
    {
        "budget_exhausted" => "WINDOW CLOSED",
        "all_lost" => "NETWORK ROLLED UP",
        _ => "OPERATION ENDED",
    };

    private async void OnRestartPressed()
    {
        try
        {
            _resolutionOverlay.Visible = false;
            _ended = false;
            _streaming = false;
            _budgetAdvisoryStamped = false;
            _state = await _client.ResetAsync();
            var firstId = SeedChannelsAndRenderFirst();
            UpdateBudgetUI();
            if (_state.Debug != null) _debugPanel?.UpdateFrom(_state.Debug);
            _debugPanel?.SetActiveAgent(firstId);
            _actionPanel.ShowLoading();
            await RefreshOptionsAsync(firstId);
        }
        catch (Exception e)
        {
            GD.PrintErr($"[Game] restart failed: {e}");
        }
    }

    private void OnQuitPressed()
    {
        GetTree().Quit();
    }

    // ── ui builder ──

    private void BuildUI()
    {
        SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);

        var theme = new Theme();
        if (HandlerFonts.Mono != null)
        {
            theme.DefaultFont = HandlerFonts.Mono;
            theme.DefaultFontSize = 13;
        }
        Theme = theme;

        var bg = new ColorRect { Color = DeskGrey };
        bg.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        AddChild(bg);

        var col = new VBoxContainer();
        col.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        col.AddThemeConstantOverride("separation", 0);
        AddChild(col);

        // Top classification banner.
        col.AddChild(BuildClassificationBanner());

        // Top bar (paper).
        var topBar = new PanelContainer();
        var topBarBg = new StyleBoxFlat
        {
            BgColor = PaperCream,
            ContentMarginLeft = 16, ContentMarginRight = 16,
            ContentMarginTop = 8, ContentMarginBottom = 8,
        };
        topBar.AddThemeStyleboxOverride("panel", topBarBg);
        col.AddChild(topBar);
        var topRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        topRow.AddThemeConstantOverride("separation", 16);
        topBar.AddChild(topRow);

        _punchcard = new BudgetPunchcard
        {
            CustomMinimumSize = new Vector2(0, 14),
            SizeFlagsVertical = SizeFlags.ShrinkCenter,
        };
        topRow.AddChild(_punchcard);

        _budgetCount = new Label { Text = "—/— DECISIONS" };
        _budgetCount.AddThemeFontSizeOverride("font_size", 13);
        _budgetCount.AddThemeColorOverride("font_color", InkBlack);
        topRow.AddChild(_budgetCount);

        topRow.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill });

        _opLabel = new Label { Text = "Loading operation…" };
        _opLabel.AddThemeFontSizeOverride("font_size", 13);
        _opLabel.AddThemeColorOverride("font_color", InkBlack);
        topRow.AddChild(_opLabel);

        // 1px hairline rule under the top bar.
        var rule = new ColorRect { Color = InkMuted, CustomMinimumSize = new Vector2(0, 1) };
        col.AddChild(rule);

        // Body.
        var body = new HBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
        body.AddThemeConstantOverride("separation", 0);
        col.AddChild(body);

        // Left: dossier tabs.
        _tabs = new DossierTabs();
        var tabsHost = new PanelContainer { CustomMinimumSize = new Vector2(180, 0) };
        var tabsBg = new StyleBoxFlat
        {
            BgColor = DeskGrey,
            ContentMarginLeft = 8, ContentMarginRight = 4,
            ContentMarginTop = 12, ContentMarginBottom = 12,
        };
        tabsHost.AddThemeStyleboxOverride("panel", tabsBg);
        tabsHost.AddChild(_tabs);
        _tabs.ChannelSelected += OnChannelSelected;
        body.AddChild(tabsHost);

        // Center: chat panel + action row.
        var center = new VBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
        center.AddThemeConstantOverride("separation", 0);
        body.AddChild(center);

        var chatPanel = new PanelContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
        var chatBg = new StyleBoxFlat
        {
            BgColor = PaperCream,
            ContentMarginLeft = 22, ContentMarginRight = 22,
            ContentMarginTop = 14, ContentMarginBottom = 14,
            BorderColor = InkMuted,
            BorderWidthLeft = 1, BorderWidthRight = 0,
            BorderWidthTop = 0, BorderWidthBottom = 0,
        };
        chatPanel.AddThemeStyleboxOverride("panel", chatBg);
        center.AddChild(chatPanel);

        _chatLog = new RichTextLabel
        {
            BbcodeEnabled = true,
            ScrollFollowing = true,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
        };
        _chatLog.AddThemeFontSizeOverride("normal_font_size", 14);
        _chatLog.AddThemeFontSizeOverride("bold_font_size", 14);
        _chatLog.AddThemeFontSizeOverride("italics_font_size", 14);
        _chatLog.AddThemeColorOverride("default_color", InkBlack);
        HandlerFonts.ApplyMonoToRichText(_chatLog);
        chatPanel.AddChild(_chatLog);

        var actionRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        actionRow.AddThemeConstantOverride("separation", 8);
        center.AddChild(actionRow);

        _actionPanel = new ActionPanel { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _actionPanel.OptionSelected += OnOptionSelected;
        actionRow.AddChild(_actionPanel);

        _stopButton = new Button
        {
            Text = "■ ABORT TX",
            CustomMinimumSize = new Vector2(120, 0),
            Visible = false,
        };
        _stopButton.Pressed += OnStopPressed;
        actionRow.AddChild(_stopButton);

        // Right column: timeline feed + (optional) debug panel.
        _rightColumn = new VBoxContainer { CustomMinimumSize = new Vector2(280, 0) };
        var rightHost = new PanelContainer();
        var rightBg = new StyleBoxFlat
        {
            BgColor = DeskGrey,
            ContentMarginLeft = 8, ContentMarginRight = 8,
            ContentMarginTop = 12, ContentMarginBottom = 12,
        };
        rightHost.AddThemeStyleboxOverride("panel", rightBg);
        rightHost.AddChild(_rightColumn);
        body.AddChild(rightHost);

        _timelineFeed = new TimelineFeed { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _rightColumn.AddChild(_timelineFeed);

        // Bottom classification banner.
        col.AddChild(BuildClassificationBanner());

        // Boot splash.
        _bootSplash = BuildSplash();
        AddChild(_bootSplash);

        // Resolution overlay.
        _resolutionOverlay = BuildResolutionOverlay();
        AddChild(_resolutionOverlay);

        _actionPanel.SetEnabled(false);
    }

    private PanelContainer BuildClassificationBanner()
    {
        var p = new PanelContainer { CustomMinimumSize = new Vector2(0, 22) };
        var sb = new StyleBoxFlat
        {
            BgColor = ClassRed,
            ContentMarginLeft = 8, ContentMarginRight = 8,
            ContentMarginTop = 2, ContentMarginBottom = 2,
        };
        p.AddThemeStyleboxOverride("panel", sb);

        var lbl = new Label
        {
            Text = "TOP SECRET // OPERATION IRON THRESHOLD // NOFORN",
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        lbl.AddThemeFontSizeOverride("font_size", 12);
        lbl.AddThemeColorOverride("font_color", new Color(0.95f, 0.93f, 0.86f));
        HandlerFonts.ApplyStencil(lbl);
        p.AddChild(lbl);
        return p;
    }

    private Control BuildSplash()
    {
        var splash = new ColorRect { Color = new Color(0.05f, 0.05f, 0.06f, 0.95f) };
        splash.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        splash.MouseFilter = MouseFilterEnum.Stop;

        var center = new CenterContainer();
        center.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        splash.AddChild(center);

        var box = new VBoxContainer();
        center.AddChild(box);

        var title = new Label
        {
            Text = "STATION BERLIN-7",
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        title.AddThemeFontSizeOverride("font_size", 32);
        title.AddThemeColorOverride("font_color", PaperCream);
        HandlerFonts.ApplyStencil(title);
        box.AddChild(title);

        _bootMessage = new Label
        {
            Text = "Booting…",
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        _bootMessage.AddThemeFontSizeOverride("font_size", 12);
        _bootMessage.AddThemeColorOverride("font_color", new Color(0.70f, 0.66f, 0.60f));
        box.AddChild(_bootMessage);

        return splash;
    }

    private void ShowBoot(string msg)
    {
        _bootSplash.Visible = true;
        _bootMessage.Text = msg;
    }

    private void HideBoot() => _bootSplash.Visible = false;

    private Control BuildResolutionOverlay()
    {
        var overlay = new ColorRect { Color = new Color(0, 0, 0, 0.85f), Visible = false };
        overlay.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        overlay.MouseFilter = MouseFilterEnum.Stop;

        var center = new CenterContainer();
        center.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        overlay.AddChild(center);

        var panel = new PanelContainer();
        var panelBg = new StyleBoxFlat
        {
            BgColor = PaperCream,
            BorderColor = InkBlack,
            BorderWidthTop = 2, BorderWidthBottom = 2, BorderWidthLeft = 2, BorderWidthRight = 2,
            ContentMarginLeft = 24, ContentMarginRight = 24,
            ContentMarginTop = 24, ContentMarginBottom = 24,
        };
        panel.AddThemeStyleboxOverride("panel", panelBg);
        center.AddChild(panel);

        var box = new VBoxContainer { CustomMinimumSize = new Vector2(640, 0) };
        box.AddThemeConstantOverride("separation", 12);
        panel.AddChild(box);

        var classification = new Label
        {
            Text = "AFTER-ACTION REPORT // CLASSIFIED",
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        classification.AddThemeFontSizeOverride("font_size", 11);
        classification.AddThemeColorOverride("font_color", ClassRed);
        HandlerFonts.ApplyStencil(classification);
        box.AddChild(classification);

        _resolutionTitle = new Label
        {
            Text = "...",
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        _resolutionTitle.AddThemeFontSizeOverride("font_size", 28);
        _resolutionTitle.AddThemeColorOverride("font_color", InkBlack);
        HandlerFonts.ApplyStencil(_resolutionTitle);
        box.AddChild(_resolutionTitle);

        _resolutionBody = new RichTextLabel
        {
            BbcodeEnabled = true,
            FitContent = true,
            ScrollActive = false,
            CustomMinimumSize = new Vector2(0, 220),
        };
        _resolutionBody.AddThemeFontSizeOverride("normal_font_size", 13);
        _resolutionBody.AddThemeFontSizeOverride("bold_font_size", 13);
        _resolutionBody.AddThemeColorOverride("default_color", InkBlack);
        HandlerFonts.ApplyMonoToRichText(_resolutionBody);
        box.AddChild(_resolutionBody);

        var buttonRow = new HBoxContainer { Alignment = BoxContainer.AlignmentMode.Center };
        buttonRow.AddThemeConstantOverride("separation", 12);
        var restart = new Button { Text = "REOPEN FILE", CustomMinimumSize = new Vector2(160, 40) };
        restart.Pressed += OnRestartPressed;
        var quit = new Button { Text = "FILE & EXIT", CustomMinimumSize = new Vector2(160, 40) };
        quit.Pressed += OnQuitPressed;
        buttonRow.AddChild(restart);
        buttonRow.AddChild(quit);
        box.AddChild(buttonRow);

        return overlay;
    }
}
