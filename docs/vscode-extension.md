# VS Code Extension

Picceler has an official VS Code extension,
[vscode-picceler](https://github.com/Robertkq/vscode-picceler), vendored as a submodule at
[`editors/vscode-picceler`](../editors/vscode-picceler).

## Getting it

1. **Install from the Extensions view** (recommended) — search **Picceler** in VS Code's Extensions
   tab and install.
2. **Manual**, e.g. to try unreleased changes:
   ```bash
   git submodule update --init --recursive
   cd editors/vscode-picceler
   npx @vscode/vsce package && code --install-extension vscode-picceler-*.vsix
   ```
   Or press `F5` in that folder for an Extension Development Host.

## Features

* Syntax highlighting for `.pic` files — types (`int64`, `float64`, `string`, `image`, `kernel`) in
  blue, builtin functions in yellow (exact shade depends on your theme).
* Hover, signature help, and snippet completion for every builtin function — e.g. typing
  `box_blur(` shows `box_blur(image img, int64 radius) -> image`.
