using Godot;

namespace Smallville;

/// <summary>
/// Centralized font loader. Two faces:
/// - <see cref="Mono"/> / <see cref="MonoBold"/> — IBM Plex Mono. Body text,
///   chat transcripts, callsign headers. Crisp typewriter feel.
/// - <see cref="Stencil"/> / <see cref="StencilBold"/> — Stardos Stencil.
///   Classification banners, decision card tags, dossier headers. Dossier feel.
///
/// Both are SIL OFL, mirrored from the Google Fonts repo. Files live at
/// godot_project/assets/fonts/. If a font isn't yet imported by the Godot
/// editor (first launch after checkout), <see cref="GD.Load"/> may return null;
/// the consumer should fall back gracefully — Godot's default font keeps
/// everything readable.
/// </summary>
public static class HandlerFonts
{
    private static FontFile? _mono;
    private static FontFile? _monoBold;
    private static FontFile? _stencil;
    private static FontFile? _stencilBold;

    public static FontFile? Mono => Ensure(ref _mono, "res://assets/fonts/IBMPlexMono-Regular.ttf");
    public static FontFile? MonoBold => Ensure(ref _monoBold, "res://assets/fonts/IBMPlexMono-Bold.ttf");
    public static FontFile? Stencil => Ensure(ref _stencil, "res://assets/fonts/StardosStencil-Regular.ttf");
    public static FontFile? StencilBold => Ensure(ref _stencilBold, "res://assets/fonts/StardosStencil-Bold.ttf");

    private static FontFile? Ensure(ref FontFile? slot, string path)
    {
        slot ??= ResourceLoader.Load<FontFile>(path, "FontFile", ResourceLoader.CacheMode.Reuse);
        return slot;
    }

    /// <summary>
    /// Set Mono as the normal font + MonoBold as the bold font on a Control.
    /// Applies via theme overrides. Falls back silently if fonts aren't loaded.
    /// </summary>
    public static void ApplyMono(Control c, string normalKey = "font", string? boldKey = null)
    {
        if (Mono != null) c.AddThemeFontOverride(normalKey, Mono);
        if (boldKey != null && MonoBold != null) c.AddThemeFontOverride(boldKey, MonoBold);
    }

    /// <summary>
    /// Apply Stardos Stencil Bold as the font on a Control. Used for
    /// classification banners, callsign headers, decision card tags, and
    /// the resolution screen title.
    /// </summary>
    public static void ApplyStencil(Control c, string key = "font", bool bold = true)
    {
        var f = bold ? StencilBold : Stencil;
        if (f != null) c.AddThemeFontOverride(key, f);
    }

    /// <summary>
    /// Apply Mono as both normal and bold font on a RichTextLabel — covers
    /// inline [b]…[/b] BBCode in the chat transcript with the bold weight.
    /// </summary>
    public static void ApplyMonoToRichText(RichTextLabel rtl)
    {
        if (Mono != null) rtl.AddThemeFontOverride("normal_font", Mono);
        if (MonoBold != null) rtl.AddThemeFontOverride("bold_font", MonoBold);
        if (Mono != null) rtl.AddThemeFontOverride("italics_font", Mono);
    }
}
