// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'Simdly',
			description: 'Cross-platform Rust library with AVX2 and NEON SIMD support for fast computations',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/mtantaoui/simdly' },
				{ icon: 'discord', label: 'Discord', href: 'https://discord.gg/your-server' },
			],
			customCss: [
				'./src/styles/custom.css',
			],
			expressiveCode: {
				themes: ['dracula', 'github-dark'],
			},
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', slug: 'introduction' },
						{ label: 'Installation', slug: 'getting-started/installation' },
						{ label: 'Quick Start', slug: 'getting-started/quick-start' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'SIMD Operations', slug: 'guides/simd-operations' },
						{ label: 'Performance Tips', slug: 'guides/performance' },
						{ label: 'Examples', slug: 'guides/example' },
					],
				},
				{
					label: 'API Reference',
					autogenerate: { directory: 'reference' },
				},
			],
			head: [
				{
					tag: 'meta',
					attrs: {
						property: 'og:image',
						// content: 'https://simdly.dev/og-image.png',
					},
				},
			],
		}),
	],
});
