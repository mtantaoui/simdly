# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the simdly crate.

## Workflows

### `publish-crate.yml` - Automated Publishing to crates.io

Automatically publishes new versions of the simdly crate to crates.io when changes are merged to the main branch.

#### Triggers

1. **Automatic (Push to main)**: 
   - Triggers on every push to the `main` branch
   - Defaults to **minor** version bump
   - Can be overridden with commit message tags

2. **Manual (Workflow Dispatch)**:
   - Can be triggered manually from GitHub Actions tab
   - Allows selection of version bump type (patch/minor/major)
   - Supports dry-run mode for testing

#### Version Bump Control

##### Automatic Version Bumping (Push to main)

- **Default**: Minor version bump (e.g., 0.1.1 → 0.2.0)
- **Override with commit messages**:
  - `[major]` - Major version bump (e.g., 0.1.1 → 1.0.0)
  - `[minor]` - Minor version bump (e.g., 0.1.1 → 0.2.0) 
  - `[patch]` - Patch version bump (e.g., 0.1.1 → 0.1.2)
  - `[no-publish]` - Skip publishing entirely

##### Manual Version Bumping (Workflow Dispatch)

- Select version bump type from dropdown: `patch`, `minor`, or `major`
- Choose dry-run mode to test without actually publishing

#### Examples

```bash
# Commit messages that trigger specific version bumps:
git commit -m "feat: add new SIMD operations [minor]"
git commit -m "fix: critical bug in AVX2 implementation [patch]"  
git commit -m "feat!: breaking API changes [major]"
git commit -m "docs: update README [no-publish]"

# Default behavior (no tag):
git commit -m "feat: add new feature"  # → minor bump
```

#### Workflow Steps

1. **Version Check**: Determines if publishing is needed and what version bump to apply
2. **Build and Test**: 
   - Code formatting check (`cargo fmt`)
   - Linting (`cargo clippy`)
   - Build with CPU feature detection
   - Run tests (including AVX2 tests if supported)
   - Generate documentation
3. **Publish**:
   - Update `Cargo.toml` version
   - Commit version bump with `[skip ci]` tag
   - Create Git tag (e.g., `v0.2.0`)
   - Publish to crates.io
   - Create GitHub release

#### Required Secrets

You need to configure these secrets in your GitHub repository settings:

1. **`CRATES_IO_TOKEN`**: Your crates.io API token
   - Go to [crates.io/me](https://crates.io/me) 
   - Generate a new token with publish permissions
   - Add it to GitHub Secrets: Settings → Secrets and variables → Actions → New repository secret

2. **`GITHUB_TOKEN`**: Automatically provided by GitHub Actions (no setup required)

#### Setup Instructions

1. **Add the crates.io token**:
   ```bash
   # In your GitHub repository:
   # Settings → Secrets and variables → Actions → New repository secret
   # Name: CRATES_IO_TOKEN
   # Value: Your token from crates.io/me
   ```

2. **Ensure proper branch protection** (optional but recommended):
   ```bash
   # Settings → Branches → Add rule for 'main'
   # ✅ Require status checks to pass before merging
   # ✅ Require branches to be up to date before merging
   ```

#### Dry Run Testing

Test the workflow without publishing:

1. Go to Actions → Publish to crates.io → Run workflow
2. Select branch: `main`
3. Version bump type: `patch` (or desired type)
4. ✅ Check "Dry run (test without publishing)"
5. Click "Run workflow"

This will run all checks and show what would be published without actually doing it.

#### Monitoring

- **Workflow status**: Check the Actions tab in your GitHub repository
- **Published versions**: Monitor at https://crates.io/crates/simdly
- **Releases**: Check GitHub Releases for automatically generated release notes

#### Troubleshooting

##### Common Issues

1. **"Version already exists"**: 
   - The version in Cargo.toml already exists on crates.io
   - Use `[no-publish]` in commit message to skip, or manually bump version

2. **"Invalid token"**: 
   - Check that `CRATES_IO_TOKEN` secret is correctly set
   - Token may have expired - generate a new one at crates.io/me

3. **"Tests failed"**: 
   - Fix any failing tests, linting issues, or formatting problems
   - The workflow won't publish if any checks fail

4. **"Permission denied"**: 
   - Ensure the GitHub token has sufficient permissions
   - Check branch protection rules aren't blocking the workflow

##### Manual Recovery

If the workflow fails partway through:

```bash
# Reset to clean state
git reset --hard HEAD~1  # Remove version bump commit if created
git tag -d v0.2.0        # Remove tag if created
git push origin :v0.2.0  # Remove tag from remote

# Then fix the issue and push again
```

#### Security Considerations

- The `CRATES_IO_TOKEN` has publish permissions - keep it secure
- The workflow only runs on `main` branch to prevent unauthorized publishes
- Version bumps are committed with `[skip ci]` to prevent infinite loops
- All changes are auditable through Git history and GitHub Actions logs