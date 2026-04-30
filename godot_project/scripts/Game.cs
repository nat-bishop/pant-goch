using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;

namespace Smallville;

public partial class Game : Control
{
    private LLMClient _client = null!;

    // ── ui ──
    private Label _actionBudgetLabel = null!;
    private Label _titleLabel = null!;
    private Label _captorNameLabel = null!;
    private RichTextLabel _captorPersonaLabel = null!;
    private RichTextLabel _chatLog = null!;
    private ActionPanel _actionPanel = null!;
    private Button _stopButton = null!;
    private HostageStatusPanel _hostagePanel = null!;
    private TimelineFeed _timelineFeed = null!;
    private DebugPanel? _debugPanel;
    private VBoxContainer _leftColumn = null!;
    private Control _bootSplash = null!;
    private Label _bootMessage = null!;
    private Control _resolutionOverlay = null!;
    private Label _resolutionTitle = null!;
    private RichTextLabel _resolutionBody = null!;

    // ── state ──
    private Scenario _scenario = new();
    private GameState _state = new();
    private CancellationTokenSource? _activeStreamCts;
    private CancellationTokenSource? _optionsCts;
    private CancellationTokenSource? _eventsCts;
    private bool _streaming;
    private bool _ended;

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
            ShowBoot("Waiting for sidecar…");
            await _client.WaitReady(180.0);

            ShowBoot("Loading scenario…");
            var scenarioTask = _client.GetScenarioAsync();
            var stateTask = _client.GetStateAsync();
            _scenario = await scenarioTask;
            _state = await stateTask;

            _titleLabel.Text = _scenario.Title;
            _captorNameLabel.Text = _scenario.Captor.Name;
            _captorPersonaLabel.Text = _scenario.Captor.Persona;
            _hostagePanel.SetHostages(_scenario.Hostages);
            _hostagePanel.UpdateFrom(_state.Hostages);
            UpdateBudgetLabel();
            MaybeMountDebugPanel();

            AppendChatSystem(_scenario.Setup);
            AppendChatSystem("Demands: " + string.Join(" / ", _scenario.Captor.InitialDemands));

            _eventsCts = new CancellationTokenSource();
            _ = _client.SubscribeEventsAsync(
                onGameEnded: OnGameEnded,
                onError: msg => GD.PrintErr($"[events] {msg}"),
                ct: _eventsCts.Token);

            HideBoot();

            // First-turn options. RefreshOptionsAsync re-enables the action panel on success.
            _actionPanel.ShowLoading();
            await RefreshOptionsAsync();
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
        _ = FireAction(hiddenLine);
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

    private void OnStopPressed()
    {
        _activeStreamCts?.Cancel();
        MarkInterrupted();
        // Cancellation discards the partial captor turn server-side, so HISTORY
        // is unchanged and the next /options call gives picks against the same
        // pre-stream state.
        RecoverFromStream();
    }

    private void OnStreamDone() => RecoverFromStream();

    private void OnStreamError(string msg)
    {
        AppendChatSystem($"[error] {msg}");
        RecoverFromStream();
    }

    private void RecoverFromStream()
    {
        SetStreamingUI(false);
        if (_ended) return;
        _actionPanel.ShowLoading();
        _ = RefreshOptionsAsync();
    }

    private async Task RefreshOptionsAsync()
    {
        _optionsCts?.Cancel();
        _optionsCts = new CancellationTokenSource();
        var ct = _optionsCts.Token;
        try
        {
            var options = await _client.GetOptionsAsync(ct);
            if (ct.IsCancellationRequested || _ended) return;
            if (options.Length == 0)
            {
                _actionPanel.ShowError("No options available.");
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
            _actionPanel.ShowError($"Options unavailable ({e.Message}).");
        }
    }

    private void ApplyStateEvent(GameState s)
    {
        _state = s;
        _hostagePanel.UpdateFrom(s.Hostages);
        UpdateBudgetLabel();
        if (s.Debug != null)
        {
            MaybeMountDebugPanel();
            _debugPanel?.UpdateFrom(s.Debug);
        }
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

    private void UpdateBudgetLabel()
    {
        var remaining = _state.ActionsRemaining;
        var max = _state.MaxActions > 0 ? _state.MaxActions : _scenario.MaxActions;
        _actionBudgetLabel.Text = max > 0
            ? $"💬 {remaining} / {max} actions left"
            : $"💬 {remaining} actions left";
        if (remaining > 0 && remaining <= 5)
        {
            _actionBudgetLabel.AddThemeColorOverride("font_color", new Color(0.95f, 0.4f, 0.4f));
        }
        else
        {
            _actionBudgetLabel.RemoveThemeColorOverride("font_color");
        }
    }

    private void MaybeMountDebugPanel()
    {
        if (_debugPanel != null) return;
        if (_state.Debug == null) return;
        _debugPanel = new DebugPanel { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _leftColumn.AddChild(_debugPanel);
        _debugPanel.UpdateFrom(_state.Debug);
    }

    private void EndGame(string reason, string outcome)
    {
        if (_ended) return;
        _ended = true;
        _activeStreamCts?.Cancel();
        _optionsCts?.Cancel();
        SetStreamingUI(false);
        _actionPanel.SetEnabled(false);

        int saved = 0, harmed = 0;
        foreach (var h in _state.Hostages)
        {
            if (h.Status == "released") saved++;
            else if (h.Status == "harmed") harmed++;
        }

        _resolutionTitle.Text = OutcomeTitle(reason);
        _resolutionBody.Clear();
        _resolutionBody.AppendText($"\n\nReason: [b]{reason}[/b]");
        _resolutionBody.AppendText($"\nCaptor outcome: [b]{outcome}[/b]");
        _resolutionBody.AppendText($"\n\nHostages saved: {saved} of {_state.Hostages.Count}");
        if (harmed > 0) _resolutionBody.AppendText($"\nHostages harmed: {harmed}");
        var max = _state.MaxActions > 0 ? _state.MaxActions : _scenario.MaxActions;
        var used = max - _state.ActionsRemaining;
        _resolutionBody.AppendText($"\nActions used: {used} of {max}");
        _resolutionOverlay.Visible = true;
    }

    private static string OutcomeTitle(string reason)
    {
        return reason switch
        {
            "captor_surrendered" => "PEACEFUL RESOLUTION",
            "all_released" => "PEACEFUL RESOLUTION",
            "hostage_harmed" => "HOSTAGE HARMED",
            "auto_breach" => "TACTICAL BREACH (forced)",
            "budget_exhausted" => "OUT OF TIME — TACTICAL BREACH",
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
            _state = await _client.ResetAsync();
            _hostagePanel.UpdateFrom(_state.Hostages);
            UpdateBudgetLabel();
            if (_state.Debug != null) _debugPanel?.UpdateFrom(_state.Debug);
            AppendChatSystem(_scenario.Setup);
            AppendChatSystem("Demands: " + string.Join(" / ", _scenario.Captor.InitialDemands));
            _actionBudgetLabel.RemoveThemeColorOverride("font_color");
            _actionPanel.ShowLoading();
            await RefreshOptionsAsync();
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
        _actionBudgetLabel = new Label { Text = "💬 — actions left" };
        _actionBudgetLabel.AddThemeFontSizeOverride("font_size", 22);
        topBar.AddChild(_actionBudgetLabel);
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
        _leftColumn = new VBoxContainer { CustomMinimumSize = new Vector2(300, 0) };
        body.AddChild(_leftColumn);
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
        _leftColumn.AddChild(captorCard);

        _timelineFeed = new TimelineFeed { SizeFlagsVertical = SizeFlags.ExpandFill };
        _leftColumn.AddChild(_timelineFeed);

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
        _actionPanel.OptionSelected += OnOptionSelected;
        actionRow.AddChild(_actionPanel);

        _stopButton = new Button
        {
            Text = "■ Stop",
            CustomMinimumSize = new Vector2(120, 0),
            Visible = false,
        };
        _stopButton.Pressed += OnStopPressed;
        actionRow.AddChild(_stopButton);

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
