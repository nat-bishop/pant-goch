export function scrollToDay(dayIndex: number) {
  document.getElementById(`day-${dayIndex}`)?.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}
