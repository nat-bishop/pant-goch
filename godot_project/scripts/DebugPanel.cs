using System.Collections.Generic;
using Godot;

namespace Smallville;

/// <summary>
/// Server-state debug overlay. Shows the three hidden meters
/// (stress / trust / desperation), the action budget, and which timeline
/// events have fired. Mounted only when the server has DEBUG_AFFORDANCES=true
/// (i.e. <see cref="GameState.Debug"/> is non-null on the first /state response).
/// </summary>
public partial class DebugPanel : PanelContainer
{
    private Label _stressLabel = null!;
    private ProgressBar _stressBar = null!;
    private Label _trustLabel = null!;
    private ProgressBar _trustBar = null!;
    private Label _despLabel = null!;
    private ProgressBar _despBar = null!;
    private Label _budgetLabel = null!;
    private Label _firedLabel = null!;

    public override void _Ready()
    {
        Build();
    }

    public void UpdateFrom(DebugView dbg)
    {
        _stressLabel.Text = $"stress     {dbg.Stress,3}";
        _stressBar.Value = dbg.Stress;
        _trustLabel.Text = $"trust       {dbg.Trust,3}";
        _trustBar.Value = dbg.Trust;
        _despLabel.Text = $"desperation {dbg.Desperation,3}";
        _despBar.Value = dbg.Desperation;
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
        header.AddThemeColorOverride("font_color", new Color(0.95f, 0.7f, 0.4f));
        col.AddChild(header);
        col.AddChild(new HSeparator());

        (_stressLabel, _stressBar) = BuildMeter(col, new Color(0.95f, 0.5f, 0.4f));
        (_trustLabel, _trustBar) = BuildMeter(col, new Color(0.5f, 0.85f, 0.6f));
        (_despLabel, _despBar) = BuildMeter(col, new Color(0.95f, 0.85f, 0.4f));

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
