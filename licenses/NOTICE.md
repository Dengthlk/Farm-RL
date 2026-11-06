# Credits

- Official game rules and Starter agent: Kaggle/kaggle-environments, distribution **1.32.7** (Apache-2.0). The bundled interpreter imports the original seed helper from a local module. A small two-player match host supplies turns and observations. See kaggle-environments-LICENSE.txt.
- Actor architecture, observation encoding, action masks and action decoding: Carson Burke, **Kaggriculture**, MIT. The game uses the CPU inference path for its two LeJEPA actors. See kaggriculture-agent-MIT.txt.
- Tiny Farm and Tiny Town pixel art: **Kenney** (CC0). See tiny-farm.txt and tiny-town.txt. The farmer motion atlas and field layout were created for this game.
- Both learned policies are shipped as actor parameters only. The original checkpoint names and SHA-256 identifiers are recorded in weights/models.json.
- The portable Windows archive includes the Python embeddable distribution, CPU PyTorch, NumPy and their runtime dependencies. Their notices accompany the packages inside runtime/. The Microsoft C++ runtime notices for the included app-local libraries are in runtime/CRT-notices/.
