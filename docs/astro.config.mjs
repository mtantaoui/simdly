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
			],
			customCss: [
				'./src/styles/custom-minimal.css',
			],
			expressiveCode: {
				themes: ['dracula', 'github-dark'],
			},
			sidebar: [
				{ label: 'Introduction', slug: 'introduction' },
				{
					label: 'Getting Started',
					items: [
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
					items: [
						{ label: 'F32x8 Vector', slug: 'reference/f32x8' },
						{ label: 'SimdLoad Trait', slug: 'reference/simd-load' },
						{ label: 'Examples', slug: 'reference/example' },
					],
				},
			],
		}),
	],
});
