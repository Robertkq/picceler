# VS Code Extension

Picceler ships an official VS Code extension, [vscode-picceler](https://github.com/Robertkq/vscode-picceler),
vendored into this repo as a git submodule at [`editors/vscode-picceler`](../editors/vscode-picceler).

**Audience:** anyone editing `.pic` files in VS Code, or updating the extension after a language
change.

## Getting it

```bash
git submodule update --init --recursive   # pulls editors/vscode-picceler in
```

To try it locally without installing anything: open `editors/vscode-picceler` as its own VS Code
workspace and press `F5` to launch an Extension Development Host with it loaded. To install it
properly, package and install the `.vsix`:

```bash
cd editors/vscode-picceler
npx @vscode/vsce package
code --install-extension vscode-picceler-*.vsix
```

See the submodule's own [HOWTO.md](https://github.com/Robertkq/vscode-picceler/blob/main/HOWTO.md)
for the full packaging/publishing process.

## What it provides

* **Syntax highlighting** (`syntaxes/picceler.tmLanguage.json`) for comments, strings, `def`/`return`,
  and:
  * **Types** (`int64`, `float64`, `string`, `image`, `kernel`) — scoped `storage.type.picceler`.
    Most themes, including VS Code's built-in Dark+, render `storage.type` in blue.
  * **Builtin functions** — scoped `support.function.builtin.{io,math,filter,algo}.picceler`. Most
    themes render `support.function` in yellow (e.g. Dark+'s `#DCDCAA`).

  Actual colors come from whichever theme the user has installed — the extension deliberately
  doesn't ship a custom theme or force `editor.tokenColorCustomizations`; it just uses the standard
  scope names most themes already color distinctly.

* **Hover, signature help, and snippet completion** for all of the builtin functions (`extension.js`),
  e.g. typing `box_blur(` shows `box_blur(image img, int64 radius) -> image`. This is a static
  lookup table, not real type-checking — it can't catch a wrong-typed argument, only show the
  expected call shape.

## Keeping it in sync with the language

The extension's builtin/type lists are hand-maintained and can drift from the compiler. The source
of truth is:

* Types: `src/lexer.cpp`'s `typeNames` list.
* Builtin functions: `MLIRGen::registerBuiltinFunctions()` in `src/mlir_gen.cpp` (the
  `_functionTable["..."]` entries) — cross-check parameter types there, since some (e.g.
  `box_blur`'s `radius`) get coerced to a specific type that isn't obvious from the call site alone.
* [`LANGUAGE.md`](../LANGUAGE.md) should already reflect both of the above; it's the easiest place
  to read the full list rather than grepping the compiler.

After a language change, update (in `editors/vscode-picceler`) `syntaxes/picceler.tmLanguage.json`'s
`types`/`builtins` patterns and the `BUILTINS` table in `extension.js`, then bump the submodule
pointer in this repo and push the extension repo itself so `git submodule update` can resolve the
new commit for other clones.
