// Pant Goch — Vellum mockup data.
//
// Same prose/glosses/scenes as Marginalia. The Vellum mockup
// reads them on a translucent parchment column laid over a
// full-viewport photograph of the current location. Glosses
// float in the photograph beside the vellum, with a red
// hairline back to the trigger paragraph.

window.WORLD = {
  player_starts_at: "the_old_vicarage",
  // The frontispiece carries an establishing shot — moor at dusk,
  // standing stones; the village as you'd arrive at it.
  frontispiece_backdrop: "../backdrops/bryn_iago_wood.png",
  locations: [
    { id: "bryn_iago_wood",       name: "Bryn Iago Wood",       xy: [44, 10], kind: "wood",     adjacent_to: ["tan_y_coed", "st_caradocs"] },
    { id: "tan_y_coed",           name: "Tan-y-coed",           xy: [20, 26], kind: "farm",     adjacent_to: ["bryn_iago_wood", "the_lane", "lower_hafod"] },
    { id: "lower_hafod",          name: "Lower Hafod",          xy: [82, 24], kind: "farm",     adjacent_to: ["tan_y_coed", "st_caradocs", "the_old_vicarage"] },
    { id: "st_caradocs",          name: "St Caradoc's",         xy: [60, 38], kind: "church",   adjacent_to: ["lower_hafod", "bryn_iago_wood", "the_post_office"] },
    { id: "the_post_office",      name: "The Post Office",      xy: [54, 50], kind: "shop",     adjacent_to: ["the_lane", "the_slaughtered_lamb", "st_caradocs"] },
    { id: "the_slaughtered_lamb", name: "The Slaughtered Lamb", xy: [44, 58], kind: "pub",      adjacent_to: ["the_lane", "the_post_office", "the_old_vicarage"] },
    { id: "the_lane",             name: "The Lane",             xy: [30, 52], kind: "junction", adjacent_to: ["tan_y_coed", "the_black_dog_bend", "the_slaughtered_lamb", "the_post_office"] },
    { id: "the_old_vicarage",     name: "The Old Vicarage",     xy: [80, 50], kind: "lodging",  adjacent_to: ["the_slaughtered_lamb", "lower_hafod"] },
    { id: "the_black_dog_bend",   name: "The Black Dog Bend",   xy: [18, 66], kind: "site",     adjacent_to: ["knighton", "the_lane"] },
    { id: "knighton",             name: "Knighton",             xy: [ 6, 74], kind: "town",     adjacent_to: ["the_black_dog_bend"] },
  ],
};

window.ROAD_PATHS = {
  "bryn_iago_wood|st_caradocs":            "M 44 11 C 50 18, 56 24, 60 36",
  "bryn_iago_wood|tan_y_coed":             "M 43 11 C 36 14, 28 18, 21 25",
  "lower_hafod|tan_y_coed":                "M 80 24 C 64 18, 46 16, 22 25",
  "lower_hafod|st_caradocs":               "M 81 25 C 74 27, 66 32, 61 37",
  "lower_hafod|the_old_vicarage":          "M 82 25 C 84 32, 84 40, 81 49",
  "st_caradocs|the_post_office":           "M 60 39 C 58 43, 56 46, 55 49",
  "the_post_office|the_slaughtered_lamb":  "M 54 51 C 50 53, 47 55, 45 57",
  "the_post_office|the_lane":              "M 53 51 C 46 50, 38 50, 31 51",
  "the_slaughtered_lamb|the_lane":         "M 43 58 C 38 56, 34 54, 31 53",
  "the_slaughtered_lamb|the_old_vicarage": "M 45 57 C 56 54, 70 52, 79 51",
  "the_lane|tan_y_coed":                   "M 29 51 C 25 44, 22 36, 21 27",
  "the_black_dog_bend|the_lane":           "M 19 65 C 22 60, 26 56, 29 53",
  "knighton|the_black_dog_bend":           "M 8 73 C 12 70, 15 68, 17 67",
};

// Glosses keyed by id. A paragraph carries `gloss_ids: [...]` and
// when that paragraph finishes streaming the glosses fade in to
// the right margin opposite that paragraph.
window.GLOSSES = {
  mrs_hughes: {
    name: "MRS HUGHES",
    body: "Landlady of the Old Vicarage. Sixty-something, widowed, sharp. Knew Megan as a child.",
  },
  iwan: {
    name: "IWAN PRITCHARD",
    body: "Megan's father. Sheep farmer at Lower Hafod, sixty. Wrote the letter that brought you here.",
  },
  megan: {
    name: "MEGAN PRITCHARD",
    body: "Died March 2019, aged 23. Single-vehicle off the road at the Black Dog Bend. Inquest: misadventure. Iwan disputes.",
    cross_ref: "see Plate I — the Bend, S-W of the village",
  },
  the_letter: {
    name: "THE LETTER",
    body: "Two pages of older hand. Iwan brought it to the vicarage himself on the Tuesday before you arrived. A line under the signature you read three times.",
  },
  black_dog_bend: {
    name: "BLACK DOG BEND",
    body: "Unmarked. B4368, two miles south-west of the village. The lay-by is recent — the council put it in after the inquest.",
  },
  old_tom: {
    name: "OLD TOM BEVAN",
    body: "Regular at the Slaughtered Lamb. Wednesdays in the corner. Drove the milk lorry on the B-roads for forty years.",
  },
  the_flowers: {
    name: "THE FLOWERS",
    body: "Plastic-wrapped. Blue ribbon, faded. Recent. Iwan brings them monthly — that much you know.",
  },
};

// Scenes — paragraphs use [phrase|gloss_id] inline markup. Each
// marked phrase renders as a red-dotted-underline hover trigger
// that opens a popup with the gloss body. The popup floats —
// it never displaces text.
window.SCENES = {
  the_old_vicarage_arrival: {
    id: "the_old_vicarage_arrival",
    location_id: "the_old_vicarage",
    location_name: "The Old Vicarage",
    chapter_no: 1,
    date_label: "Monday — Arrival",
    backdrop: "../backdrops/the_old_vicarage.png",
    paragraphs: [
      { text: "Three hours west of London the road forgets it was ever a motorway, then forgets it was ever wide, then forgets, in a hedge-tunnel of blackthorn and rain, that it was a road at all." },
      { text: "I arrived at the Old Vicarage in the last of the daylight. [Mrs Hughes|mrs_hughes] was in the hall with a fire-iron and a small, exact smile." },
      { text: "She handed me a single key on a brass ring. \"[Letter|the_letter] for you on the table, my love. From [Iwan|iwan]. He brought it Tuesday.\" She did not say more. She went back to the kitchen to lay a fire I could already hear catching." },
      { text: "On the hall table: two careful pages of older hand. An address. A phone number. A line under the signature that I read three times before I took my coat off." },
    ],
    options: [
      { text: "Open the letter properly. Read it as he wrote it.", kind: "engage" },
      { text: "Ring the number now. Before the courage wears off.", kind: "engage" },
      { text: "Walk down to Lower Hafod in the morning.", kind: "leave", leaves_for: "lower_hafod" },
      { text: "Down to the village first. See the place before the man.", kind: "leave", leaves_for: "the_slaughtered_lamb" },
    ],
  },

  lower_hafod_doorstep: {
    id: "lower_hafod_doorstep",
    location_id: "lower_hafod",
    location_name: "Lower Hafod",
    chapter_no: 2,
    date_label: "Tuesday — Iwan's Door",
    backdrop: "../backdrops/lower_hafod.png",
    paragraphs: [
      { text: "The lane to Lower Hafod is single-track for the last half mile, hedged so high the car feels swallowed. I rounded a final bend and the farmhouse was there, stone the colour of wet slate, smoke pulling sideways out of the chimney into the wind." },
      { text: "I parked where the gravel widened. The dogs did not bark." },
      { text: "[Iwan Pritchard|iwan] is taller than I expected, and not as old. Sixty, perhaps. He looked at me for a long moment in the doorway before he stepped back into the hall." },
      { text: "\"Mr—,\" he said. \"You came. I'll put the kettle on. Sit there, if you would.\"" },
      { text: "The hall was cold and the kitchen was warm. He set the cup down in front of me and sat opposite and did not speak until the clock in the front room had gone off the half hour. \"[Megan|megan],\" he said. \"You'll be wanting to hear about Megan.\"" },
    ],
    options: [
      { text: "\"Thank you for writing. I'd like to start with what you wrote.\"", kind: "engage" },
      { text: "Sit. Let the silence work.", kind: "intuit" },
      { text: "Ask, gently, about the flowers at the bend.", kind: "risk", note: "you don't know yet that it's him" },
      { text: "Back to the village. The pub at lunchtime.", kind: "leave", leaves_for: "the_slaughtered_lamb" },
    ],
  },

  slaughtered_lamb_tom: {
    id: "slaughtered_lamb_tom",
    location_id: "the_slaughtered_lamb",
    location_name: "The Slaughtered Lamb",
    chapter_no: 3,
    date_label: "Wednesday — The Public Bar",
    backdrop: "../backdrops/slaughtered_lamb.png",
    paragraphs: [
      { text: "The bar at the Slaughtered Lamb is half-full of farmers and a fire that does most of the work. A small dog at the hearth turned its head when I came in and decided not to mind." },
      { text: "[Old Tom Bevan|old_tom] was at the corner table where he is on Wednesdays. Brains Bitter, three-quarters full. Cap on the table beside him. He had watched me order without looking at me do it." },
      { text: "When I stood at his table he did not invite me to sit, but he did not tell me to leave." },
      { text: "\"You're not from round here,\" he said. \"I can see that.\"" },
    ],
    options: [
      { text: "\"I'm a journalist. Iwan Pritchard asked me to come.\"", kind: "engage" },
      { text: "Say nothing yet. Ask if you can buy him another.", kind: "intuit" },
      { text: "Sit anyway. Wait for him.", kind: "risk" },
      { text: "Out the back. Walk the lane.", kind: "leave", leaves_for: "the_lane" },
    ],
  },

  black_dog_bend_alone: {
    id: "black_dog_bend_alone",
    location_id: "the_black_dog_bend",
    location_name: "The Black Dog Bend",
    chapter_no: 4,
    date_label: "Thursday — Last Light",
    backdrop: "../backdrops/black_dog_bend.png",
    paragraphs: [
      { text: "There is no sign for the [Black Dog Bend|black_dog_bend]. I drove past it the first time. Pulled in twenty yards before, in the lay-by, where the verge falls steeply away into a hedge you cannot see across." },
      { text: "Out of the car the wind was something else. Not loud. Just continuous, and from a direction the body had to lean against." },
      { text: "The [flowers|the_flowers] were there. Plastic-wrapped, faded blue ribbon, recent. Below them the grass grew differently — paler, finer, in a patch the shape of nothing in particular." },
      { text: "I didn't see, at first, where the Mini went over. I saw it on the third look, and only because I stopped trying to." },
    ],
    options: [
      { text: "Stand here. Let the place tell you what it tells you.", kind: "intuit", note: "time will pass" },
      { text: "Look at the flowers more carefully. The ribbon. The wrapping.", kind: "engage" },
      { text: "Back to the lane. Down into the village.", kind: "leave", leaves_for: "the_lane" },
    ],
  },
};

window.SCENE_FOR_LOCATION = {
  the_old_vicarage:     "the_old_vicarage_arrival",
  lower_hafod:          "lower_hafod_doorstep",
  the_black_dog_bend:   "black_dog_bend_alone",
  the_slaughtered_lamb: "slaughtered_lamb_tom",
};
