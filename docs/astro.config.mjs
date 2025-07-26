// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'simdly',
			description: 'High-performance Rust library leveraging SIMD for fast computations',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/mtantaoui/simdly' },
			],
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
					],
				},
				{
					label: 'API Reference',
					autogenerate: { directory: 'reference' },
				},
			],
		}),
	],
});
