using System.Collections.Generic;
using System.Linq;
using Godot;

namespace Smallville;

/// <summary>
/// Server-state debug overlay. Per-channel meters (stress / trust) for the
/// active dossier, plus the action budget and which timeline events have
/// fired. Mounted only when the server returns DEBUG_AFFORDANCES=true (i.e.
/// <see cref="HandlerState.Debug"/> is non-null on the first /state response).
/// </summary>
public partial class DebugPanel : PanelContainer
{
    private Label _activeAgentLabel = null!;
    private Label _stressLabel = null!;
    private ProgressBar _stressBar = null!;
    private Label _trustLabel = null!;
    private ProgressBar _trustBar = null!;
    private Label _budgetLabel = null!;
    private Label _firedLabel = null!;

    private string _activeAgentId = "";
    private DebugView? _lastDebug;

    public override void _Ready()
    {
        Build();
    }

    public void SetActiveAgent(string agentId)
    {
        _activeAgentId = agentId;
        if (_lastDebug != null) UpdateFrom(_lastDebug);
    }

    public void UpdateFrom(DebugView dbg)
    {
        _lastDebug = dbg;
        var a = dbg.Agents.FirstOrDefault(x => x.Id == _activeAgentId);
        if (a == null && dbg.Agents.Count > 0) a = dbg.Agents[0];

        if (a != null)
        {
            _activeAgentLabel.Text = $"// {a.Id.ToUpperInvariant()}";
            _stressLabel.Text = $"stress {a.Stress,3}";
            _stressBar.Value = a.Stress;
            _trustLabel.Text = $"trust  {a.Trust,3}";
            _trustBar.Value = a.Trust;
        }
        _budgetLabel.Text = $"actions: {dbg.ActionCount} / {dbg.MaxActions}";
        _firedLabel.Text = dbg.FiredTimelineEventIds.Count == 0
            ? "fired: —"
            : $"fired: {string.Join(", ", dbg.FiredTimelineEventIds)}";
    }

    private void Build()
    {
        var col = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        col.AddThemeConstantOverride("separation", 4);
        AddChild(col);

        var header = new Label { Text = "DEBUG" };
        header.AddThemeFontSizeOverride("font_size", 12);
        header.AddThemeColorOverride("font_color", new Color(0.83f, 0.65f, 0.20f));
        col.AddChild(header);
        col.AddChild(new HSeparator());

        _activeAgentLabel = new Label { Text = "// —" };
        _activeAgentLabel.AddThemeFontSizeOverride("font_size", 11);
        _activeAgentLabel.AddThemeColorOverride("font_color", new Color(0.6f, 0.6f, 0.6f));
        col.AddChild(_activeAgentLabel);

        (_stressLabel, _stressBar) = BuildMeter(col, new Color(0.95f, 0.5f, 0.4f));
        (_trustLabel, _trustBar) = BuildMeter(col, new Color(0.5f, 0.85f, 0.6f));

        _budgetLabel = new Label { Text = "actions: 0 / 0" };
        _budgetLabel.AddThemeFontSizeOverride("font_size", 11);
        col.AddChild(_budgetLabel);

        _firedLabel = new Label { Text = "fired: —" };
        _firedLabel.AddThemeFontSizeOverride("font_size", 11);
        _firedLabel.AddThemeColorOverride("font_color", new Color(0.6f, 0.6f, 0.6f));
        col.AddChild(_firedLabel);
    }

    private static (Label, ProgressBar) BuildMeter(VBoxContainer parent, Color color)
    {
        var label = new Label { Text = "—" };
        label.AddThemeFontSizeOverride("font_size", 11);
        parent.AddChild(label);

        var bar = new ProgressBar
        {
            MinValue = 0,
            MaxValue = 100,
            ShowPercentage = false,
            CustomMinimumSize = new Vector2(0, 6),
        };
        bar.AddThemeColorOverride("font_color", color);
        parent.AddChild(bar);

        return (label, bar);
    }
}
