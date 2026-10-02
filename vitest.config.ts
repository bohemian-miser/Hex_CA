import { defineConfig } from 'vitest/config';

// The CA tests run whole boards to quiet; the budget is a Raspberry Pi (DESIGN.md §8).
export default defineConfig({
  test: {
    testTimeout: 60_000,
    hookTimeout: 60_000,
  },
});
