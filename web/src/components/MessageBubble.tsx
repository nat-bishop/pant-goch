import clsx from "clsx";

export type BubblePosition = "single" | "first" | "middle" | "last";
export type BubbleVariant = "me" | "them";

interface Props {
  body: string;
  variant: BubbleVariant;
  position?: BubblePosition;
}

const RADIUS_FULL = "1rem";
const RADIUS_JOINED = "0.25rem";

// Authored prose (welcome, plot beats) often arrives with hard wraps from a
// YAML block scalar. Inside a chat bubble those look like a poem. Collapse
// single line breaks to spaces; preserve double breaks as paragraph gaps.
function flowParagraphs(body: string): string {
  return body.replace(/([^\n])\n([^\n])/g, "$1 $2");
}

function radii(variant: BubbleVariant, position: BubblePosition) {
  const own = variant === "me";
  const top = position === "first" || position === "single";
  const bot = position === "last" || position === "single";
  return {
    borderTopLeftRadius: own || top ? RADIUS_FULL : RADIUS_JOINED,
    borderTopRightRadius: !own || top ? RADIUS_FULL : RADIUS_JOINED,
    borderBottomLeftRadius: own || bot ? RADIUS_FULL : RADIUS_JOINED,
    borderBottomRightRadius: !own || bot ? RADIUS_FULL : RADIUS_JOINED,
  };
}

export function MessageBubble({
  body,
  variant,
  position = "single",
}: Props) {
  const own = variant === "me";

  return (
    <div
      className={clsx(
        "flex w-full flex-col animate-bubble-in",
        own ? "items-end" : "items-start",
      )}
    >
      <div
        className={clsx(
          "max-w-[min(560px,75%)] whitespace-pre-line break-words px-3.5 py-2 text-[14px] leading-[1.45]",
          own
            ? "bg-accent text-text-on-accent"
            : "bg-bg-bubble-them text-text",
        )}
        style={radii(variant, position)}
      >
        {flowParagraphs(body)}
      </div>
    </div>
  );
}
