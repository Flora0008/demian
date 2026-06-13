# Project website (GitHub Pages)

The DeMiAn project page is served from this `docs/` folder by GitHub Pages
(Jekyll). It is self-contained — `_layouts/project_demian.html` defines the whole
page (hero from the front-matter + theme CSS); no external Jekyll theme is needed.

```
docs/
├── _config.yml                 # minimal Jekyll config (set baseurl for a project page)
├── index.md                    # the project page (front matter + body; the DeMiAn write-up)
├── _layouts/project_demian.html # self-contained layout (hero, CSS, content, footer)
└── assets/
    ├── img/demian/*.png         # figures
    └── video/demian/*.mp4       # carousel (example_01–10) + comparison clip (~140 MB total)
```

## Enable it

1. Push the repo to GitHub.
2. **Settings → Pages → Build and deployment → Source: "Deploy from a branch",
   Branch: `main` (or your default), folder: `/docs`.**
3. Set `baseurl` in `docs/_config.yml`:
   - Project page (`https://<org>.github.io/<repo>/`) → `baseurl: "/<repo>"`.
   - User/org page (`https://<org>.github.io/`) → leave `baseurl: ""`.
   (The page/layout reference assets via `relative_url`, which prepends `baseurl`.)
4. Set `code_url` in `index.md`'s front matter to this repo's URL (it powers the
   hero "Code" button); `paper_url` is already set.

The page then renders at `https://<org>.github.io/<repo>/`.

## Notes

- All referenced assets are committed here (largest video 24 MB, under the 100 MB
  GitHub file limit; ~140 MB total). To keep the repo lean you can instead host the
  videos on a CDN/HF and repoint the carousel `BASE` URL + the `<video src>` in
  `index.md`.
- Source: adapted from `pearls-lab.github.io/_projects/demian.md` (the lab-site
  collection page) — the "← All projects" link and lab footer were dropped for the
  standalone repo page.
