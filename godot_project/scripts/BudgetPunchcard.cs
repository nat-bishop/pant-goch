using Godot;

namespace Smallville;

/// <summary>
/// Top-bar budget visualization. A row of small dots — filled for spent
/// decisions, hollow rings for remaining. Custom-drawn via _Draw() so a
/// scenario with N=18 actions doesn't require 18 child nodes.
///
/// No color shift on low budget — the diegetic "BUDGET ADVISORY" stamp into
/// the chat log handles low-budget pressure (see Game.cs).
/// </summary>
public partial class BudgetPunchcard : Control
{
    private static readonly Color FilledColor = new(0.10f, 0.10f, 0.11f);   // ink black
    private static readonly Color HollowColor = new(0.40f, 0.38f, 0.36f);   // ink muted

    private const float DotRadius = 4.0f;
    private const float DotSpacing = 11.0f;

    private int _max = 0;
    private int _used = 0;

    public void Set(int used, int max)
    {
        if (used == _used && max == _max) return;
        _used = Mathf.Clamp(used, 0, max);
        _max = Mathf.Max(0, max);
        CustomMinimumSize = new Vector2(_max * DotSpacing + 4, 12);
        QueueRedraw();
    }

    public override void _Draw()
    {
        for (int i = 0; i < _max; i++)
        {
            var pos = new Vector2(DotRadius + 2 + i * DotSpacing, Size.Y / 2);
            if (i < _used)
            {
                DrawCircle(pos, DotRadius, FilledColor);
            }
            else
            {
                // hollow: ring effect via two circles (outer fill, inner background-cream).
                // We draw an outer dark ring and an inner cream cutout so the dot looks
                // hollow without depending on the parent panel's exact background color.
                DrawCircle(pos, DotRadius, HollowColor);
                DrawCircle(pos, DotRadius - 1.5f, new Color(0.94f, 0.90f, 0.82f)); // PaperCream
            }
        }
    }
}
