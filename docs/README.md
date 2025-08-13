# Simdly Documentation

[![Built with Starlight](https://astro.badg.es/v2/built-with-starlight/tiny.svg)](https://starlight.astro.build)

This is the official documentation website for **Simdly**, a cross-platform Rust library that provides high-performance SIMD operations with support for both AVX2 (x86/x86_64) and NEON (ARM) instruction sets.

## About Simdly

Simdly is a Rust crate that offers:
- 🚀 Cross-platform SIMD support (AVX2 + NEON)
- 🛡️ Safe abstractions over unsafe SIMD operations
- ⚡ Zero-cost abstractions with compile-time optimizations
- 🔧 Universal API that works across different architectures

## 🔗 Links

- **Crate Repository**: [github.com/mtantaoui/simdly](https://github.com/mtantaoui/simdly)
- **Crates.io**: [crates.io/crates/simdly](https://crates.io/crates/simdly)
- **Documentation**: This website

## 🚀 Documentation Structure

This documentation site is built with Astro + Starlight and includes:

```
src/content/docs/
├── getting-started/
│   ├── installation.md
│   └── quick-start.md
├── guides/
│   ├── example.md
│   ├── performance.md
│   └── simd-operations.md
├── index.mdx
└── introduction.md
```

The documentation covers installation, usage examples, performance benchmarks, and complete API reference for the Simdly crate.

## 🧞 Development Commands

To work on this documentation site locally:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:4321`      |
| `npm run build`           | Build your production site to `./dist/`          |
| `npm run preview`         | Preview your build locally, before deploying     |

## 🎨 Theme

This documentation features a custom space-themed design inspired by HAL 9000 from "2001: A Space Odyssey", with:
- Animated starfield background
- Space-black color scheme
- Glowing HAL 9000 eye imagery
- Cross-platform SIMD focus

## 📚 Contributing

To contribute to the Simdly crate documentation:
1. Fork the [main repository](https://github.com/mtantaoui/simdly)
2. Update documentation in the `docs/` directory
3. Test locally with `npm run dev`
4. Submit a pull request

For the Rust crate itself, see the main repository's contributing guidelines.
