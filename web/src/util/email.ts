// "Janet Rivera <jrivera@stmaria-hospital.org>" → "Janet Rivera"
export function stripAddress(field: string): string {
  if (!field) return field;
  const lt = field.indexOf("<");
  return lt > 0 ? field.slice(0, lt).trim() : field.trim();
}

// game_time_hours starts at 0.0 (= "Day 1, 8:00 AM" by display convention).
// Each hour of game-time advances the wall clock by an hour. Returns a string
// like "Day 1, 9:30 AM".
export function formatGameTime(hours: number): string {
  const STARTING_HOUR_OF_DAY = 8;
  const wallClockHours = hours + STARTING_HOUR_OF_DAY;
  const day = Math.floor(wallClockHours / 24) + 1;
  const hourOfDay = Math.floor(wallClockHours % 24);
  const minutes = Math.floor((hours % 1) * 60);
  const ampm = hourOfDay >= 12 ? "PM" : "AM";
  const displayHour =
    hourOfDay === 0 ? 12 : hourOfDay > 12 ? hourOfDay - 12 : hourOfDay;
  const minStr = minutes.toString().padStart(2, "0");
  return `Day ${day}, ${displayHour}:${minStr} ${ampm}`;
}
