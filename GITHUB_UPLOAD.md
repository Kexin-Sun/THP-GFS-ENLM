# Upload this archive to GitHub

Run these commands from the extracted `THP-GFS-ENLM` repository root.
Upload the extracted directory with Git, rather than committing its ZIP file.
The included `.gitignore` keeps caches and new runs/reports out of version
control; all frozen results and archive summaries remain included.
The `.gitattributes` file disables newline conversion for every frozen file.

## Check the archive

```bash
python verify_frozen_results.py
git --version
```

To regenerate the existing reports without training:

```bash
bash regenerate_reports.sh
```

## Initialize a new local repository

For a fresh extracted directory without an existing Git history:

```bash
git init -b main
git add .
git status --short
git commit -m "Archive all five THP-GFS-ENLM experimental groups"
```

If Git asks for author identity, set your own name and email locally and repeat
the commit. Replace both placeholders:

```bash
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL"
```

## Create and push using GitHub CLI

With `gh` installed, authenticate and create a private repository:

```bash
gh auth login
gh repo create THP-GFS-ENLM --private --source=. --remote=origin --push
```

Use `--public` instead of `--private` if you intend public access.
Do not run both this route and the manual route below for the same upload.

## Push to a manually created empty repository

Alternatively, create an empty GitHub repository named `THP-GFS-ENLM`.
Do not initialize the remote with a README, license, or `.gitignore`, since
the local archive already has its own first commit. Configure command-line
authentication, replace `YOUR_USERNAME`, and run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/THP-GFS-ENLM.git
git remote -v
git push -u origin main
```

If this directory already has Git history or a configured origin, inspect
`git status` and `git remote -v` before applying the fresh-repository commands.
These instructions do not require overwriting any remote history.

## Verify a clone

After uploading, replace `YOUR_USERNAME` and check an independent clone:

```bash
git clone https://github.com/YOUR_USERNAME/THP-GFS-ENLM.git THP-GFS-ENLM-check
cd THP-GFS-ENLM-check
python verify_frozen_results.py
```

The verification must report all 1004 original result files as unchanged.

## References

- [GitHub: adding locally hosted code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [GitHub CLI: gh repo create](https://cli.github.com/manual/gh_repo_create)
- [GitHub: file size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
