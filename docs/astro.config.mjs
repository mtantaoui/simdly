// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://mtantaoui.github.io/simdly',
	integrations: [
		starlight({
			title: 'Simdly',
			description: 'Cross-platform Rust library with AVX2 and NEON SIMD support for fast computations',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/mtantaoui/simdly' },
			],
			head: [],
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
					label: 'API Reference',
					link: 'https://docs.rs/simdly',
				},
			],
		}),
	],
});
