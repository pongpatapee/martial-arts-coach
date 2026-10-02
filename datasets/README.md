# Clip dataset

One folder per clip under `clips/`. Add clips with `sparring clip add`; don't create folders by
hand (it computes the video's hash and metadata). See "Sample footage" in the top-level README.

```
clips/<clip-id>/
  video.<ext>    git-ignored; restore from your backup if missing (`sparring clip check`)
  clip.toml      committed
  labels.json    committed; written by `sparring label <clip-id>`
```

## clip.toml

| Field | Meaning |
|---|---|
| `discipline` | `boxing`, `kickboxing` or `muay_thai`: selects the move set and detectors |
| `split` | `train` or `test`. Test clips are never used by `sparring train` by default |
| `recorded` | Recording date |
| `camera` | Camera setup, e.g. `side-on, chest height`, `angled from the corner` |
| `source` | Where the footage came from, and that the people in it agreed to its use |
| `notes` | Anything else worth knowing (lighting, a southpaw, mostly clinch) |
| `video`, `sha256` | The video file and its hash. Labels are tied to these exact frames |
| `fps`, `duration_s`, `width`, `height` | Measured from the video |

Edit the descriptive fields freely. Don't edit `video` or `sha256`, and don't re-encode or trim
the video: add the new version as a new clip.

## Ids

`<discipline>-<place>-<date>[-r<round>]`, lowercase letters, digits and hyphens, e.g.
`boxing-homegym-2026-10-04-r2`.
