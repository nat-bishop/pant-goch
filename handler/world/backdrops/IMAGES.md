# Pant Goch — backdrop manifest

Reference document for the location backdrops + map served by `handler/world/`. Each entry has a stable shape so an asset-generation script can parse it.

---

## Global specs

| Spec | Value |
|---|---|
| **Source-of-truth resolution** | **2560 × 1440** (16:9 PNG, sRGB, 8-bit) |
| **Map resolution** | **2816 × 1536** (1.833:1, matches `map.png` native aspect) |
| **Shipping path** | `handler/world/backdrops/{location_id}.png` and `handler/world/map.png` |
| **Shipping resolution** | downsample to **1408 × 768** for the live frontend (kept as a separate file or re-export) |
| **Aspect of the displayed pane** | varies — `background-size: cover` crops dynamically; **compose subject in the centre 1:1** so it survives a near-square crop |
| **CSS treatment applied at display time** | `brightness(0.78) saturate(0.78) contrast(1.04) sepia(0.06)` + grain overlay + vignette — author warm and slightly saturated; the runtime mutes it |

## Global aesthetic

Folk-noir investigation set in the Welsh Marches (Welsh-English borderlands). Late winter / early spring, March/April 2026. The game's tonal touchstones are Jon McGregor's *Reservoir 13*, Benjamin Myers' *The Gallows Pole*, Mark Jenkin's *Enys Men*, and Jem Southam's landscape photography.

- **Time of day:** dusk or overcast late-afternoon. Pewter overcast or last-light gold-grey. **Never midday sun.**
- **Palette:** wet-slate greys, ochres, sodden greens, blackthorn-and-rain. Muted but not desaturated at source (the CSS desaturates).
- **No people in any image.** Places only.
- **Welsh Marches specificity, not generic English countryside.** Stone farmhouses with slate roofs, single-track lanes hedged tall, drystone walls, sheep on the hills, occasional bilingual road signs.
- **Restraint over drama.** Documentary register, not cinematic. No HDR, no fish-eye, no dramatic skies, no captions/watermarks/frames.

---

## Map

### `map.png`

- **Status:** ✓ have *(reauthor optional, at 2816 × 1536)*
- **Resolution (target):** 2816 × 1536
- **Subject:** hand-drawn top-down parish map of Pant Goch and surroundings.
- **Composition:** flat-art top-down survey map; pewter background, faded ochres, sepia ink. Ten labelled locations placed at their canonical fractional coordinates (see `world/map.yaml`). The frontend overlays clickable hotspot pins on top.
- **Required props:** all ten location names rendered as ink-labels; the B4368 visible; the River Teme visible; rough hill shading.
- **Avoid:** photographic style; modern cartographic conventions; people; vehicles.
- **Prompt seed:** *A hand-drawn antiquarian parish map of a fictional Welsh-Marches village, top-down survey style, pewter parchment background, faded sepia ink lines, gentle ochre hill shading, no scale bar, no compass rose. Bilingual place names in serif lettering. Style of a 19th-century estate map.*

---

## Location backdrops (10)

Files are named `{location_id}.png` exactly as listed below. Names must match the `id` field in `handler/world/locations/*.yaml`.

### `the_old_vicarage.png`

- **Status:** ✓ have *(reauthor optional)*
- **Subject:** the player's lodgings — a small late-Victorian stone vicarage (now a B&B) on the eastern edge of the village. The player arrives at dusk on day one.
- **Composition:** the house seen from the lane that approaches it. Hedge in foreground, dusk light catching the windows. Slate roof, stone walls, smoke from the chimney, a single car parked askew on gravel.
- **Required props:** stone Victorian vicarage; lit interior windows; trail of chimney smoke.
- **Avoid:** people; modern signage; bright daylight.
- **Prompt seed:** *A small late-Victorian stone vicarage in the Welsh Marches at dusk, slate roof, smoke trailing sideways from the chimney, one warm-lit window, a gravel driveway with a single car parked askew, approached down a high-hedged single-track lane. Overcast pewter sky, last gold-grey light, no people. Documentary register, restrained, in the style of a still from Mark Jenkin's Enys Men.*

### `lower_hafod.png`

- **Status:** ✓ have *(reauthor optional)*
- **Subject:** Iwan and Wendy Pritchard's working sheep farm, easternmost in the parish. A grief-tired house.
- **Composition:** approach view across a yard of compacted gravel, the lane's bend visible behind. The house seen end-on, "stone the colour of wet slate". Smoke trailing horizontally out of the chimney in the wind.
- **Required props:** stone slate-roofed farmhouse; trailing chimney smoke pulled sideways by wind; muddy gravel yard; high hedge.
- **Avoid:** people; barking dogs; bright weather; toys or signs of children.
- **Prompt seed:** *A working Welsh-Marches sheep farm at overcast late afternoon, single-track lane bending past, "stone the colour of wet slate" farmhouse with smoke pulling horizontally sideways out of the chimney in a strong wind, compacted gravel yard, no people, no dogs. Bare hedgerows, late winter mud, drained pewter light. Documentary style, the loneliness of a grief-tired house.*

### `tan_y_coed.png`

- **Status:** ✗ to author
- **Subject:** Eluned Vaughn's working sheep farm. Imposing, guarded. The kitchen is rarely opened to strangers; visitors are received in the yard.
- **Composition:** the gated yard from the lane outside, or the farmhouse with its outbuildings. **A Land Rover** (Defender or Series, dark colour, mud-flecked) must be visible — parked, half-glimpsed through a gate, or behind a barn. The whole place reads as guarded.
- **Required props:** **dark mud-flecked Land Rover Defender** in the yard (LOAD-BEARING); closed gate; stone outbuildings; working-farm signage of feed bins and water troughs; sheep dogs absent on camera but felt off-frame.
- **Avoid:** people; visible sheep dogs (the dogs are off-camera, only sensed); friendly open gates; warm light.
- **Prompt seed:** *A working Welsh-Marches sheep farm seen from the lane outside the closed yard gate at overcast late afternoon, stone outbuildings, a dark mud-flecked Land Rover Defender parked in the yard half-glimpsed behind the gate, no people, no visible dogs but a sense of dogs nearby, late-winter mud, drained pewter light. A guarded, hostile-feeling place. Documentary register.*

### `the_slaughtered_lamb.png`

- **Status:** ✓ have *(filename renamed from `slaughtered_lamb.png`)*
- **Subject:** the village pub, seen from inside. Half-full of farmers and a fire that does most of the work. Real ale (Brains Bitter).
- **Composition:** interior view of a small two-room Welsh-Marches pub looking toward the bar and hearth. Low beamed ceiling, smoke-dark walls, a coal or log fire alive in a stone hearth doing most of the lighting. Brass-and-pump-clip bar with hand-pulls for Brains Bitter, a row of upturned glasses, a single low-watt pendant lamp. Wooden settles and a couple of round tables, empty pint glasses left on one. A leaded window in the far wall showing only blue-black dusk outside. No people, but the room reads as recently occupied — chairs pulled out, a folded local newspaper on a table.
- **Required props:** stone hearth with a live fire (the principal light source); bar with brass hand-pull beer engines and a Brains Bitter pump-clip; low beamed ceiling; dark wooden settles or pews; upturned glasses on a shelf behind the bar; a leaded window showing dusk outside; no people but signs of recent presence (half-drained pint, folded newspaper, pulled-out chair).
- **Avoid:** modern pub-chain branding; flatscreens or fruit machines; bright overhead lighting; tourists or people in shot; gastro-pub aesthetics (chalkboard menus of small plates, exposed-bulb chandeliers); cocktails; clean-edge modern renovation.
- **Prompt seed:** *Interior of a small Welsh-Marches village pub on a Wednesday evening, low beamed ceiling, smoke-dark walls, a live coal-and-log fire in a stone hearth doing most of the lighting, a short bar with brass hand-pull beer engines and a Brains Bitter pump-clip, upturned glasses on a shelf behind, wooden settles and round tables, a half-drained pint and a folded local newspaper left on one table, a leaded window in the far wall showing blue-black dusk, no people but the room recently occupied, warm umber and ember-orange lighting against deep shadow, late winter. Documentary register, the atmosphere of a working-farmer's local — Mark Jenkin's Enys Men by way of a folk-noir interior.*

### `the_post_office.png`

- **Status:** ✓ have *(filename renamed from `post_office.png`)*
- **Subject:** Maud Pryce's shop — the village post office and informal gossip hub.
- **Composition:** the shop seen from the lane outside, hand-painted "POST OFFICE / SIOP" bilingual signage above the window, a small grocery aisle visible through the door, a noticeboard with parish notices and lost-dog flyers next to the entrance. Stone terrace of cottages adjoining.
- **Required props:** bilingual painted signage (English + Welsh — "SIOP"); a turning postcard rack visible through the window; parish noticeboard outside.
- **Avoid:** people; modern branded signage; ATM machines; cars in front.
- **Prompt seed:** *A small Welsh village post office at overcast dusk, hand-painted bilingual "POST OFFICE / SIOP" signage above leaded windows, parish noticeboard with handwritten lost-dog flyers next to the door, a turning postcard rack glimpsed through the window, stone terrace of cottages on either side, no people, no cars. Documentary register, late winter.*

### `st_caradocs.png`

- **Status:** ✗ to author
- **Subject:** the parish church. A small ancient Welsh-Marches stone church (St Caradoc is a 12th-century Welsh hermit-saint). Father Gareth Hughes conducted Megan's funeral here.
- **Composition:** the churchyard at dusk, viewed from the lych-gate or down the path between yews. Headstones in foreground, church tower against an overcast sky. Optional: a single weathered bouquet on a recent grave.
- **Required props:** ancient stone church with a square or short tower; old yew trees; lych-gate; leaning headstones; low stone churchyard wall.
- **Avoid:** people; rooks too prominent; modern grave decorations; floodlights; obvious recent renovation.
- **Prompt seed:** *A small ancient Welsh-Marches parish church at overcast dusk, square stone tower, weathered yew trees, a lych-gate in the foreground, leaning lichen-covered headstones, low drystone churchyard wall, damp grass, no people, late winter. Documentary register, the silence of a country churchyard before evensong.*

### `the_lane.png`

- **Status:** ✗ to author
- **Subject:** the village three-way junction — where the lane up to Tan-y-coed meets the road down past the Black Dog Bend toward Knighton, and the branch east back into Pant Goch (church, pub, post office). A junction the player traverses many times.
- **Composition:** standing slightly back from the junction so all three branches of the fingerpost are legible. The fingerpost is in the centre-right of the frame at roughly head height, weathered black-and-white, three arms angled to match the three branches of the road. Camera looks down the south-west branch (the Knighton road, the road Megan died on) — it curves away into hedge and wooded distance. Hedgerow detail close on both sides, single-track tarmac with grass growing through the centre, no road markings, light failing.
- **Required props:** **a black-and-white wooden fingerpost with exactly three weathered destination arms**, lettering legible and reading: (1) **"TAN-Y-COED ½"** pointing roughly upward-left (north, uphill), (2) **"KNIGHTON 4 / TREF-Y-CLAWDD"** pointing down the south-west branch — bilingual, two lines, (3) **"PANT GOCH ¼"** pointing back to the east toward the village; high hedgerow (blackthorn, hazel — bare-branched); single-track tarmac with central grass strip; no road markings; no street lighting.
- **Avoid:** painted road markings (no white lines); tarmac in good repair (it should look badly resurfaced and going to pieces); modern reflective road signs; brown-tourist signs; people; cars; the wrong number of arms on the fingerpost; misspelled or invented place names — only the three names above should appear.
- **Prompt seed:** *A rural Welsh-Marches three-way junction at failing dusk light, a weathered black-and-white wooden fingerpost with three arms in the centre-right of the frame: one arm angled up-left reading "TAN-Y-COED ½", one arm angled down the lane reading "KNIGHTON 4 / TREF-Y-CLAWDD" (bilingual, two lines), one arm pointing back behind the camera-right reading "PANT GOCH ¼". The camera looks down the Knighton branch — a single-track tarmac lane with grass growing through the centre, badly resurfaced, no road markings, high blackthorn hedges bare-branched on both sides, curving away into wooded distance. Overcast pewter sky, no cars, no people, late winter. Documentary register, the loneliness of a country junction the night before something happens.*

### `the_black_dog_bend.png`

- **Status:** ✓ have *(strongly recommend reauthoring at higher resolution — this image carries the most weight in the game)*
- **Subject:** the crime scene. The unmarked bend on the B4368 where Megan Pritchard went off the road on the night of 14 March 2019. Iwan brings a single bunch of plastic-wrapped flowers once a month; the bunch in the image is the most recent — about a week old.
- **Composition:** the **bend** is the subject. Wide framing of the curve itself, viewed from the lay-by twenty yards back. The road curves away to the left, the steep verge drops into hedge you cannot see across, no signage marks the place. The grief-tokens are deliberately small and easy to miss on first look: at the base of a single fence-post pushed into the verge, a **small weathered wooden cross** (driftwood-grey, no more than a foot tall, lichen-flecked, the two pieces lashed at the joint, no name, no plaque) — almost reading as part of the fence on a casual glance. Tucked against the foot of the cross, **one small bunch of plastic-wrapped supermarket flowers** about a week old: petals beginning to weather (browned edges, drooping but not collapsed), plastic dulled and faintly water-spotted, a faded blue ribbon still tied. Nothing else in the foreground — no pile, no drift, no graveyard of bouquets. Just the cross and the one bunch.
- **Required props:** unmarked tarmac curve as the principal subject; small gravel lay-by twenty yards back; steep verge falling into impenetrable bare-branched hedgerow; **a single small weathered wooden cross** (driftwood-grey, ~12 inches, lashed joint, lichen, easy to miss); **one small plastic-wrapped supermarket bouquet about a week old** at the foot of the cross — weathered but not decayed, faded blue ribbon, dulled plastic; no signage; no road markings.
- **Avoid:** **piles or drifts of flowers** (only one bunch, recent); fresh-looking bouquets (these are a week weathered); fully-decayed/rotted flowers (Iwan would have removed them by now); multiple crosses or any cross larger than a foot; named plaques, photographs, candles, teddy bears, or any other elaborate roadside-shrine paraphernalia; police tape; warning signs; people; bright weather; obvious crash debris (it's been seven years); commercial florist arrangements (this is a £4 forecourt bunch).
- **Prompt seed:** *An unmarked tarmac bend on a rural Welsh-Marches B-road at drained late-winter dusk, the road curving away to the left into a steep verge that drops into impenetrable bare-branched hedgerow, a small gravel lay-by twenty yards back. The bend itself is the subject. Almost unnoticed at the base of a single fence-post pushed into the verge: a small weathered wooden cross about a foot tall, two pieces of driftwood-grey timber lashed at the joint, lichen-flecked, no plaque. At the foot of the cross, a single small plastic-wrapped supermarket bouquet about a week old — petals browning at the edges and drooping but not collapsed, plastic dulled and water-spotted, a faded blue ribbon. Nothing else — no pile of older bouquets, no other tributes. Overcast pewter sky, no signage, no people, no cars. The image carries grief by under-stating it. Documentary register, restraint above all — the eye should land on the bend first and only notice the cross on a second look.*

### `bryn_iago_wood.png`

- **Status:** ✓ have *(reauthor optional)*
- **Subject:** a small ancient wood on the northern ridge. Folkloric weight (Bryn Iago — Welsh "Iago hill"). Dylan retreats here; Megan met him here in 2018. The kind of place secrets are kept.
- **Composition:** inside the wood looking through the trees. A path winding away between bare-branched oaks and beeches. Wet leaves underfoot. Mossy stones. Late-afternoon, the light failing through the canopy.
- **Required props:** bare-branched mature deciduous trees (oak, beech); a path; moss on stone or bark; wet ground; no people.
- **Avoid:** conifers; well-maintained signage; benches; clearings with views; bright weather.
- **Prompt seed:** *Inside an ancient Welsh-Marches wood in late winter at failing afternoon light, bare-branched oak and beech, a narrow earth path winding away between mossy stones, wet fallen leaves, no people, no signage, no benches, the light filtering pewter through the canopy. Folkloric atmosphere, the quiet of a place where secrets are kept. Documentary register.*

### `knighton.png`

- **Status:** ✗ to author
- **Subject:** the nearest market town — Knighton (Tref-y-clawdd), where the regional CID is based. DC Helen Mortimer will only meet the player at the Costa, not at the station.
- **Composition:** the high street at overcast late afternoon, looking down toward a clocktower at the end. Parked cars; stone shopfronts; bilingual road signs visible. Less remote than the village but still small, hills visible at the end of the street.
- **Required props:** stone-built Welsh market-town shopfronts; a clocktower at the perspective vanishing point; bilingual road sign somewhere (English + Welsh); a Costa or similar small chain visible but understated; hills visible at the end of the street.
- **Avoid:** people; busy traffic; bright sun; pure-pedestrian zone; obviously English (Cotswolds-pretty) architecture.
- **Prompt seed:** *The main high street of a small Welsh-border market town at overcast late afternoon, looking down the street toward a Victorian stone clocktower at the end, stone-built shopfronts with weathered hand-painted fascias, a bilingual road sign, a small Costa coffee storefront understated among the others, parked cars, hills visible at the end of the street, no people walking, late winter, drained light. Documentary register.*

---

## Optional: frontispiece cover

### `cover.png` *(optional)*

- **Status:** ✗ optional add
- **Subject:** the cover image shown on the left pane while the title page sits on the right. Establishes the game's tone before the player starts.
- **Composition:** a wide view of the Welsh Marches at dusk — bare moor or fields rising to a ridge, a single road dropping into the valley toward the village (implied, not seen), last light catching a standing stone or distant farmhouse.
- **Required props:** wide landscape vista; a single road or path leading away; ridge or hill horizon; suggestion of distance and lateness of hour.
- **Avoid:** people; cars; the village itself in view; dramatic sunsets; lens flare.
- **Prompt seed:** *A wide vista of the Welsh Marches at the last of the daylight, bare moor and field rising to a long ridge, a single tarmac road descending into a valley toward an unseen village, last gold-grey light catching the topmost stone of an isolated standing stone in the middle distance, overcast above, no people, no cars, late winter. The opening shot of a 90-minute folk-horror film. Documentary register, restraint.*
