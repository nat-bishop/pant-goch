using Godot;

namespace Smallville;

/// <summary>
/// Four-button action UI: Offer (with submenu), Refuse, Empathize, Threaten.
/// Builds its children programmatically. Emits ActionSelected(type, detail).
/// </summary>
public partial class ActionPanel : PanelContainer
{
    [Signal] public delegate void ActionSelectedEventHandler(string type, string detail);

    private VBoxContainer _root = null!;
    private HBoxContainer _mainRow = null!;
    private GridContainer _offerSubmenu = null!;
    private Button _offerBtn = null!;
    private bool _offerExpanded = false;
    private bool _enabled = true;

    public override void _Ready()
    {
        Build();
    }

    public void SetEnabled(bool enabled)
    {
        _enabled = enabled;
        ApplyEnabled(_root);
    }

    private void ApplyEnabled(Node n)
    {
        foreach (var child in n.GetChildren())
        {
            if (child is Button b)
            {
                b.Disabled = !_enabled;
            }
            else
            {
                ApplyEnabled(child);
            }
        }
    }

    private void Build()
    {
        _root = new VBoxContainer
        {
            CustomMinimumSize = new Vector2(0, 110),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        AddChild(_root);

        var label = new Label { Text = "Your move:" };
        _root.AddChild(label);

        _mainRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _root.AddChild(_mainRow);

        _offerBtn = MakeBtn("Offer ▾", () => ToggleOfferSubmenu());
        _mainRow.AddChild(_offerBtn);
        _mainRow.AddChild(MakeBtn("Refuse", () => Emit("refuse", "")));
        _mainRow.AddChild(MakeBtn("Empathize", () => Emit("empathize", "")));
        _mainRow.AddChild(MakeBtn("Threaten", () => Emit("threaten", "")));

        _offerSubmenu = new GridContainer
        {
            Columns = 4,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            Visible = false,
        };
        _offerSubmenu.AddChild(MakeBtn("→ Vehicle", () => Emit("offer", "vehicle")));
        _offerSubmenu.AddChild(MakeBtn("→ Money", () => Emit("offer", "money")));
        _offerSubmenu.AddChild(MakeBtn("→ Safe passage", () => Emit("offer", "safe_passage")));
        _offerSubmenu.AddChild(MakeBtn("→ Medical", () => Emit("offer", "medical")));
        _root.AddChild(_offerSubmenu);
    }

    private Button MakeBtn(string text, System.Action onPressed)
    {
        var b = new Button
        {
            Text = text,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(0, 40),
        };
        b.Pressed += () => onPressed();
        return b;
    }

    private void ToggleOfferSubmenu()
    {
        if (!_enabled) return;
        _offerExpanded = !_offerExpanded;
        _offerSubmenu.Visible = _offerExpanded;
        _offerBtn.Text = _offerExpanded ? "Offer ▴" : "Offer ▾";
    }

    private void Emit(string type, string detail)
    {
        if (!_enabled) return;
        if (_offerExpanded)
        {
            _offerExpanded = false;
            _offerSubmenu.Visible = false;
            _offerBtn.Text = "Offer ▾";
        }
        EmitSignal(SignalName.ActionSelected, type, detail);
    }
}
