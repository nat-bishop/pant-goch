export function scrollToCycle(cycleIndex: number) {
  document.getElementById(`cycle-${cycleIndex}`)?.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}
