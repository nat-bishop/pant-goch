using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;

namespace Smallville;

public partial class Game : Control
{
    private LLMClient _client = null!;

    // ── ui ──
    private Label _timerLabel = null!;
    private Label _titleLabel = null!;
    private Label _captorNameLabel = null!;
    private RichTextLabel _captorPersonaLabel = null!;
    private RichTextLabel _chatLog = null!;
    private ActionPanel _actionPanel = null!;
    private Button _stopButton = null!;
    private Button _regenButton = null!;
    private HostageStatusPanel _hostagePanel = null!;
    private TimelineFeed _timelineFeed = null!;
    private Control _bootSplash = null!;
    private Label _bootMessage = null!;
    private Control _resolutionOverlay = null!;
    private Label _resolutionTitle = null!;
    private RichTextLabel _resolutionBody = null!;

    // ── state ──
    private Scenario _scenario = new();
    private GameState _state = new();
    private CancellationTokenSource? _activeStreamCts;
    private CancellationTokenSource? _eventsCts;
    private bool _streaming;
    private bool _ended;
    private double _timeLeftSeconds;
    private int _displayedSecond = -1;

    // ── lifecycle ──

    public override void _Ready()
    {
        _client = GetNode<LLMClient>("/root/LLMClient");
        BuildUI();
        ShowBoot("Booting…");
        _ = BootAsync();
    }

    public override void _Process(double delta)
    {
        if (_streaming || _ended || _scenario.DurationSeconds == 0) return;
        _timeLeftSeconds = Mathf.Max(0, _timeLeftSeconds - delta);
        UpdateTimerLabel();
        if (_timeLeftSeconds <= 0 && !_ended)
        {
            EndGame("timer_expired", "breached");
        }
    }

    public override void _ExitTree()
    {
        _activeStreamCts?.Cancel();
        _eventsCts?.Cancel();
    }

    // ── boot ──

    private async Task BootAsync()
    {
        try
        {
            ShowBoot("Waiting for sidecar…");
            await _client.WaitReady(180.0);

            ShowBoot("Loading scenario…");
            _scenario = await _client.GetScenarioAsync();
            _state = await _client.GetStateAsync();

            _titleLabel.Text = _scenario.Title;
            _captorNameLabel.Text = _scenario.Captor.Name;
            _captorPersonaLabel.Text = _scenario.Captor.Persona;
            _hostagePanel.SetHostages(_scenario.Hostages);
            _hostagePanel.UpdateFrom(_state.Hostages);

            _timeLeftSeconds = _state.TimeRemainingSeconds > 0
                ? _state.TimeRemainingSeconds
                : _scenario.DurationSeconds;
            UpdateTimerLabel();

            AppendChatSystem(_scenario.Setup);
            AppendChatSystem("Demands: " + string.Join(" / ", _scenario.Captor.InitialDemands));

            _eventsCts = new CancellationTokenSource();
            _ = _client.SubscribeEventsAsync(
                onTimelineEvent: OnTimelineEvent,
                onGameEnded: OnGameEnded,
                onError: msg => GD.PrintErr($"[events] {msg}"),
                ct: _eventsCts.Token);

            HideBoot();

            // Debug regenerate button
            var debug = (System.Environment.GetEnvironmentVariable("DEBUG_AFFORDANCES") ?? "").ToLower();
            _regenButton.Visible = debug == "1" || debug == "true" || debug == "yes";

            EnableActionInput();
        }
        catch (Exception e)
        {
            GD.PrintErr($"[Game] boot failed: {e}");
            ShowBoot($"Boot failed: {e.Message}\n\nIs the sidecar running? Is llama-server up?");
        }
    }

    // ── action flow ──

    private void OnActionSelected(string line)
    {
        if (_ended) return;
        _ = FireAction(line);
    }

    private async Task FireAction(string line)
    {
        _activeStreamCts?.Cancel();
        _activeStreamCts = new CancellationTokenSource();
        var ct = _activeStreamCts.Token;

        SetStreamingUI(true);
        AppendChatPlayerLine(line);
        BeginCaptorBubble();

        try
        {
            await _client.StreamActionAsync(
                line,
                onTextDelta: AppendCaptorDelta,
                onState: ApplyStateEvent,
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

    private void OnStopPressed()
    {
        _activeStreamCts?.Cancel();
        MarkInterrupted();
        SetStreamingUI(false);
        EnableActionInput();
    }

    private void OnRegenPressed()
    {
        if (_ended) return;
        _activeStreamCts?.Cancel();
        _activeStreamCts = new CancellationTokenSource();
        var ct = _activeStreamCts.Token;
        SetStreamingUI(true);
        AppendChatSystem("(regenerating…)");
        BeginCaptorBubble();
        _ = _client.RegenerateAsync(
            onTextDelta: AppendCaptorDelta,
            onState: ApplyStateEvent,
            onDone: OnStreamDone,
            onError: OnStreamError,
            ct: ct);
    }

    private void OnStreamDone()
    {
        SetStreamingUI(false);
        EnableActionInput();
    }

    private void OnStreamError(string msg)
    {
        AppendChatSystem($"[error] {msg}");
        SetStreamingUI(false);
        EnableActionInput();
    }

    private void EnableActionInput()
    {
        _actionPanel.SetEnabled(true);
        _actionPanel.GrabInputFocus();
    }

    private void ApplyStateEvent(GameState s)
    {
        _state = s;
        _hostagePanel.UpdateFrom(s.Hostages);
        if (s.TimeRemainingSeconds > 0) _timeLeftSeconds = s.TimeRemainingSeconds;
        if (s.Ended)
        {
            EndGame(s.EndReason ?? "ended", s.CaptorOutcome ?? "ended");
        }
    }

    private void OnTimelineEvent(TimelineEvent evt)
    {
        _timelineFeed.AddEvent(evt);
    }

    private void OnGameEnded(string reason, string outcome)
    {
        EndGame(reason, outcome);
    }

    // ── chat log helpers ──

    private void AppendChatPlayerLine(string line)
    {
        _chatLog.AppendText("\n\n");
        _chatLog.PushColor(new Color(0.5f, 0.85f, 0.95f));
        _chatLog.PushBold();
        _chatLog.AppendText("Negotiator: ");
        _chatLog.Pop();
        _chatLog.AppendText(line);
        _chatLog.Pop();
    }

    private void AppendChatSystem(string text)
    {
        _chatLog.AppendText("\n\n");
        _chatLog.PushColor(new Color(0.55f, 0.55f, 0.55f));
        _chatLog.PushItalics();
        _chatLog.AppendText(text);
        _chatLog.Pop();
        _chatLog.Pop();
    }

    private void BeginCaptorBubble()
    {
        _chatLog.AppendText("\n\n");
        _chatLog.PushColor(new Color(0.95f, 0.65f, 0.5f));
        _chatLog.PushBold();
        _chatLog.AppendText($"{_scenario.Captor.Name}: ");
        _chatLog.Pop();
        _chatLog.Pop();
    }

    private void AppendCaptorDelta(string delta)
    {
        _chatLog.AppendText(delta);
    }

    private void MarkInterrupted()
    {
        _chatLog.PushColor(new Color(0.7f, 0.5f, 0.5f));
        _chatLog.PushItalics();
        _chatLog.AppendText("  [interrupted]");
        _chatLog.Pop();
        _chatLog.Pop();
    }

    // ── ui state ──

    private void SetStreamingUI(bool streaming)
    {
        _streaming = streaming;
        if (streaming) _actionPanel.SetEnabled(false);
        _stopButton.Visible = streaming;
    }

    private void UpdateTimerLabel()
    {
        var t = (int)Math.Ceiling(_timeLeftSeconds);
        if (t == _displayedSecond) return;
        _displayedSecond = t;
        _timerLabel.Text = $"⏱  {t / 60:D2}:{t % 60:D2}";
        if (t < 60) _timerLabel.AddThemeColorOverride("font_color", new Color(0.95f, 0.4f, 0.4f));
    }

    private void EndGame(string reason, string outcome)
    {
        if (_ended) return;
        _ended = true;
        _activeStreamCts?.Cancel();
        SetStreamingUI(false);
        _actionPanel.SetEnabled(false);

        int saved = 0, harmed = 0;
        foreach (var h in _state.Hostages)
        {
            if (h.Status == "released") saved++;
            else if (h.Status == "harmed") harmed++;
        }

        _resolutionTitle.Text = OutcomeTitle(reason, outcome);
        _resolutionBody.Clear();
        _resolutionBody.AppendText($"\n\nReason: [b]{reason}[/b]");
        _resolutionBody.AppendText($"\nCaptor outcome: [b]{outcome}[/b]");
        _resolutionBody.AppendText($"\n\nHostages saved: {saved} of {_state.Hostages.Count}");
        if (harmed > 0) _resolutionBody.AppendText($"\nHostages harmed: {harmed}");
        _resolutionBody.AppendText($"\nTime elapsed: {(_scenario.DurationSeconds - (int)_timeLeftSeconds) / 60} min");
        _resolutionOverlay.Visible = true;
    }

    private static string OutcomeTitle(string reason, string outcome)
    {
        return reason switch
        {
            "captor_surrendered" => "PEACEFUL RESOLUTION",
            "all_released" => "PEACEFUL RESOLUTION",
            "hostage_harmed" => "HOSTAGE HARMED",
            "auto_breach" => "TACTICAL BREACH (forced)",
            "timer_expired" => "TIME UP — TACTICAL BREACH",
            _ => "SCENARIO ENDED",
        };
    }

    private async void OnRestartPressed()
    {
        try
        {
            _resolutionOverlay.Visible = false;
            _ended = false;
            _streaming = false;
            _chatLog.Clear();
            await _client.ResetAsync();
            _state = await _client.GetStateAsync();
            _hostagePanel.UpdateFrom(_state.Hostages);
            _timeLeftSeconds = _scenario.DurationSeconds;
            _displayedSecond = -1;
            UpdateTimerLabel();
            AppendChatSystem(_scenario.Setup);
            AppendChatSystem("Demands: " + string.Join(" / ", _scenario.Captor.InitialDemands));
            _timerLabel.RemoveThemeColorOverride("font_color");
            EnableActionInput();
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

        var bg = new ColorRect { Color = new Color(0.10f, 0.10f, 0.12f) };
        bg.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        AddChild(bg);

        var margin = new MarginContainer();
        margin.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);
        margin.AddThemeConstantOverride("margin_left", 12);
        margin.AddThemeConstantOverride("margin_right", 12);
        margin.AddThemeConstantOverride("margin_top", 8);
        margin.AddThemeConstantOverride("margin_bottom", 8);
        AddChild(margin);

        var col = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        margin.AddChild(col);

        // top bar
        var topBar = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        col.AddChild(topBar);
        _timerLabel = new Label { Text = "⏱  00:00" };
        _timerLabel.AddThemeFontSizeOverride("font_size", 22);
        topBar.AddChild(_timerLabel);
        topBar.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill });
        _titleLabel = new Label { Text = "Loading…" };
        _titleLabel.AddThemeFontSizeOverride("font_size", 18);
        topBar.AddChild(_titleLabel);

        col.AddChild(new HSeparator());

        // body
        var body = new HBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
        body.AddThemeConstantOverride("separation", 12);
        col.AddChild(body);

        // left
        var left = new VBoxContainer { CustomMinimumSize = new Vector2(300, 0) };
        body.AddChild(left);
        var captorCard = new PanelContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        var captorBox = new VBoxContainer();
        captorCard.AddChild(captorBox);
        var captorHeader = new Label { Text = "Captor" };
        captorHeader.AddThemeFontSizeOverride("font_size", 14);
        captorHeader.AddThemeColorOverride("font_color", new Color(0.6f, 0.6f, 0.6f));
        captorBox.AddChild(captorHeader);
        _captorNameLabel = new Label { Text = "..." };
        _captorNameLabel.AddThemeFontSizeOverride("font_size", 20);
        captorBox.AddChild(_captorNameLabel);
        _captorPersonaLabel = new RichTextLabel
        {
            FitContent = true,
            ScrollActive = false,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            BbcodeEnabled = false,
            CustomMinimumSize = new Vector2(0, 110),
        };
        _captorPersonaLabel.AddThemeFontSizeOverride("normal_font_size", 12);
        captorBox.AddChild(_captorPersonaLabel);
        left.AddChild(captorCard);

        _timelineFeed = new TimelineFeed { SizeFlagsVertical = SizeFlags.ExpandFill };
        left.AddChild(_timelineFeed);

        // center
        var center = new VBoxContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
        body.AddChild(center);

        var chatPanel = new PanelContainer
        {
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsVertical = SizeFlags.ExpandFill,
        };
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
        chatPanel.AddChild(_chatLog);

        var actionRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        center.AddChild(actionRow);

        _actionPanel = new ActionPanel { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _actionPanel.ActionSelected += OnActionSelected;
        actionRow.AddChild(_actionPanel);

        _stopButton = new Button
        {
            Text = "■ Stop",
            CustomMinimumSize = new Vector2(120, 0),
            Visible = false,
        };
        _stopButton.Pressed += OnStopPressed;
        actionRow.AddChild(_stopButton);

        _regenButton = new Button
        {
            Text = "↻ Regen",
            CustomMinimumSize = new Vector2(80, 0),
            Visible = false,
        };
        _regenButton.Pressed += OnRegenPressed;
        actionRow.AddChild(_regenButton);

        // right
        var right = new VBoxContainer { CustomMinimumSize = new Vector2(300, 0) };
        body.AddChild(right);
        _hostagePanel = new HostageStatusPanel { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        right.AddChild(_hostagePanel);

        // boot splash
        _bootSplash = BuildSplash();
        AddChild(_bootSplash);

        // resolution overlay
        _resolutionOverlay = BuildResolutionOverlay();
        AddChild(_resolutionOverlay);

        _actionPanel.SetEnabled(false);
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
            Text = "SMALLVILLE",
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        title.AddThemeFontSizeOverride("font_size", 32);
        box.AddChild(title);

        _bootMessage = new Label
        {
            Text = "Booting…",
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        _bootMessage.AddThemeFontSizeOverride("font_size", 14);
        _bootMessage.AddThemeColorOverride("font_color", new Color(0.7f, 0.7f, 0.7f));
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
        center.AddChild(panel);

        var margin = new MarginContainer();
        margin.AddThemeConstantOverride("margin_left", 24);
        margin.AddThemeConstantOverride("margin_right", 24);
        margin.AddThemeConstantOverride("margin_top", 24);
        margin.AddThemeConstantOverride("margin_bottom", 24);
        panel.AddChild(margin);

        var box = new VBoxContainer { CustomMinimumSize = new Vector2(560, 0) };
        box.AddThemeConstantOverride("separation", 12);
        margin.AddChild(box);

        _resolutionTitle = new Label
        {
            Text = "...",
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        _resolutionTitle.AddThemeFontSizeOverride("font_size", 26);
        box.AddChild(_resolutionTitle);

        _resolutionBody = new RichTextLabel
        {
            BbcodeEnabled = true,
            FitContent = true,
            ScrollActive = false,
            CustomMinimumSize = new Vector2(0, 160),
        };
        _resolutionBody.AddThemeFontSizeOverride("normal_font_size", 14);
        box.AddChild(_resolutionBody);

        var buttonRow = new HBoxContainer { Alignment = BoxContainer.AlignmentMode.Center };
        var restart = new Button { Text = "Restart", CustomMinimumSize = new Vector2(140, 40) };
        restart.Pressed += OnRestartPressed;
        var quit = new Button { Text = "Quit", CustomMinimumSize = new Vector2(140, 40) };
        quit.Pressed += OnQuitPressed;
        buttonRow.AddChild(restart);
        buttonRow.AddChild(quit);
        box.AddChild(buttonRow);

        return overlay;
    }
}
