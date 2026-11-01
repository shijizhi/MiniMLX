# Release versions

MiniMLX uses three-component versions such as `0.1.0`. The Rust workspace,
Python package and runtime must match. Check them with:

```sh
.venv/bin/python tools/version.py
```

To prepare a new version, update all version entries together:

```sh
.venv/bin/python tools/version.py 0.2.0
cargo metadata --locked --offline --no-deps --format-version 1 >/dev/null
```

Review and commit the version changes alongside the release changes. Create the
matching Git tag (for example, `v0.2.0`) when publishing. Version changes do not
automatically create commits, tags or releases.

The packaging script checks version consistency before building. It writes the
same version into the app's Info.plist and the DMG filename and refuses to
overwrite an existing DMG. Signed builds are named `MiniMLX-0.2.0-arm64.dmg`;
ad-hoc test builds contain `-test`. The notarization script defaults to the
signed DMG for the current version and updates its SHA-256 after stapling.

Changing a version does not modify previously built or notarized DMGs. Rebuild,
sign, notarize and test a new image for each release.
